import matplotlib
matplotlib.use('Agg')
import os, sys
import yaml
from argparse import ArgumentParser
from tqdm import tqdm

import imageio
import numpy as np
from skimage.transform import resize
from skimage import img_as_ubyte
import torch
import torch.nn.functional as F
from sync_batchnorm import DataParallelWithCallback

from modules.generator import OcclusionAwareGenerator, OcclusionAwareSPADEGenerator
from modules.keypoint_detector import KPDetector, HEEstimator
from animate import normalize_kp
from scipy.spatial import ConvexHull


if sys.version_info[0] < 3:
    raise Exception("You must use Python 3 or higher. Recommended version is Python 3.7")

def load_checkpoints(config_path, checkpoint_path, gen, cpu=False):

    with open(config_path) as f:
        config = yaml.load(f)

    if gen == 'original':
        generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                            **config['model_params']['common_params'])
    elif gen == 'spade':
        generator = OcclusionAwareSPADEGenerator(**config['model_params']['generator_params'],
                                                 **config['model_params']['common_params'])

    if not cpu:
        generator.cuda()

    kp_detector = KPDetector(**config['model_params']['kp_detector_params'],
                             **config['model_params']['common_params'])
    if not cpu:
        kp_detector.cuda()

    he_estimator = HEEstimator(**config['model_params']['he_estimator_params'],
                               **config['model_params']['common_params'])
    if not cpu:
        he_estimator.cuda()
    
    if cpu:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_path)
 
    generator.load_state_dict(checkpoint['generator'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    he_estimator.load_state_dict(checkpoint['he_estimator'])
    
    if not cpu:
        generator = DataParallelWithCallback(generator)
        kp_detector = DataParallelWithCallback(kp_detector)
        he_estimator = DataParallelWithCallback(he_estimator)

    generator.eval()
    kp_detector.eval()
    he_estimator.eval()
    
    return generator, kp_detector, he_estimator


def headpose_pred_to_degree(pred):
    device = pred.device
    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).to(device)
    pred = F.softmax(pred)
    degree = torch.sum(pred*idx_tensor, axis=1) * 3 - 99

    return degree

'''
# beta version
def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    roll_mat = torch.cat([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll), 
                          torch.zeros_like(roll), torch.cos(roll), -torch.sin(roll),
                          torch.zeros_like(roll), torch.sin(roll), torch.cos(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    pitch_mat = torch.cat([torch.cos(pitch), torch.zeros_like(pitch), torch.sin(pitch), 
                           torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch),
                           -torch.sin(pitch), torch.zeros_like(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), -torch.sin(yaw), torch.zeros_like(yaw),  
                         torch.sin(yaw), torch.cos(yaw), torch.zeros_like(yaw),
                         torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', roll_mat, pitch_mat, yaw_mat)

    return rot_mat

'''
def get_rotation_matrix(yaw, pitch, roll):
    yaw = yaw / 180 * 3.14
    pitch = pitch / 180 * 3.14
    roll = roll / 180 * 3.14

    roll = roll.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    yaw = yaw.unsqueeze(1)

    pitch_mat = torch.cat([torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
                          torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
                          torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1)
    pitch_mat = pitch_mat.view(pitch_mat.shape[0], 3, 3)

    yaw_mat = torch.cat([torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
                           torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
                           -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1)
    yaw_mat = yaw_mat.view(yaw_mat.shape[0], 3, 3)

    roll_mat = torch.cat([torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
                         torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
                         torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1)
    roll_mat = roll_mat.view(roll_mat.shape[0], 3, 3)

    rot_mat = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)

    return rot_mat

def keypoint_transformation(kp_canonical, he, estimate_jacobian=True, free_view=False, yaw=0, pitch=0, roll=0):
    kp = kp_canonical['value']
    if not free_view:
        yaw, pitch, roll = he['yaw'], he['pitch'], he['roll']
        yaw = headpose_pred_to_degree(yaw)
        pitch = headpose_pred_to_degree(pitch)
        roll = headpose_pred_to_degree(roll)
    else:
        if yaw is not None:
            yaw = torch.tensor([yaw]).cuda()
        else:
            yaw = he['yaw']
            yaw = headpose_pred_to_degree(yaw)
        if pitch is not None:
            pitch = torch.tensor([pitch]).cuda()
        else:
            pitch = he['pitch']
            pitch = headpose_pred_to_degree(pitch)
        if roll is not None:
            roll = torch.tensor([roll]).cuda()
        else:
            roll = he['roll']
            roll = headpose_pred_to_degree(roll)

    t, exp = he['t'], he['exp']

    rot_mat = get_rotation_matrix(yaw, pitch, roll)
    
    # keypoint rotation
    kp_rotated = torch.einsum('bmp,bkp->bkm', rot_mat, kp)

    # keypoint translation
    t = t.unsqueeze_(1).repeat(1, kp.shape[1], 1)
    kp_t = kp_rotated + t

    # add expression deviation 
    exp = exp.view(exp.shape[0], -1, 3)
    kp_transformed = kp_t + exp

    if estimate_jacobian:
        jacobian = kp_canonical['jacobian']
        jacobian_transformed = torch.einsum('bmp,bkps->bkms', rot_mat, jacobian)
    else:
        jacobian_transformed = None

    return {'value': kp_transformed, 'jacobian': jacobian_transformed}

def make_animation(source_image, driving_video, generator, kp_detector, he_estimator, relative=True, adapt_movement_scale=True, estimate_jacobian=True, cpu=False, free_view=False, yaw=0, pitch=0, roll=0):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_image[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()
        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_canonical = kp_detector(source)
        he_source = he_estimator(source)
        he_driving_initial = he_estimator(driving[:, :, 0])

        kp_source = keypoint_transformation(kp_canonical, he_source, estimate_jacobian)
        kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, estimate_jacobian)
        # kp_driving_initial = keypoint_transformation(kp_canonical, he_driving_initial, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()
            he_driving = he_estimator(driving_frame)
            kp_driving = keypoint_transformation(kp_canonical, he_driving, estimate_jacobian, free_view=free_view, yaw=yaw, pitch=pitch, roll=roll)
            kp_norm = normalize_kp(kp_source=kp_source, kp_driving=kp_driving,
                                   kp_driving_initial=kp_driving_initial, use_relative_movement=relative,
                                   use_relative_jacobian=estimate_jacobian, adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)

            predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
    return predictions

def find_best_frame(source, driving, cpu=False):
    import face_alignment

    def normalize_kp(kp):
        kp = kp - kp.mean(axis=0, keepdims=True)
        area = ConvexHull(kp[:, :2]).volume
        area = np.sqrt(area)
        kp[:, :2] = kp[:, :2] / area
        return kp

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=True,
                                      device='cpu' if cpu else 'cuda')
    kp_source = fa.get_landmarks(255 * source)[0]
    kp_source = normalize_kp(kp_source)
    norm  = float('inf')
    frame_num = 0
    for i, image in tqdm(enumerate(driving)):
        kp_driving = fa.get_landmarks(255 * image)[0]
        kp_driving = normalize_kp(kp_driving)
        new_norm = (np.abs(kp_source - kp_driving) ** 2).sum()
        if new_norm < norm:
            norm = new_norm
            frame_num = i
    return frame_num

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='config/vox-256.yaml', help="path to config")
    parser.add_argument("--checkpoint", default='', help="path to checkpoint to restore")

    parser.add_argument("--source_image", default='', help="path to source image")
    parser.add_argument("--driving_video", default='', help="path to driving video")
    parser.add_argument("--result_video", default='', help="path to output")

    parser.add_argument("--gen", default="spade", choices=["original", "spade"])
 
    parser.add_argument("--relative", dest="relative", action="store_true", help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true", help="adapt movement scale based on convex hull of keypoints")

    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true", 
                        help="Generate from the frame that is the most alligned with source. (Only for faces, requires face_aligment lib)")

    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None,  
                        help="Set frame to start from.")
 
    parser.add_argument("--cpu", dest="cpu", action="store_true", help="cpu mode.")

    parser.add_argument("--free_view", dest="free_view", action="store_true", help="control head pose")
    parser.add_argument("--yaw", dest="yaw", type=int, default=None, help="yaw")
    parser.add_argument("--pitch", dest="pitch", type=int, default=None, help="pitch")
    parser.add_argument("--roll", dest="roll", type=int, default=None, help="roll")
 

    parser.set_defaults(relative=False)
    parser.set_defaults(adapt_scale=False)
    parser.set_defaults(free_view=False)

    opt = parser.parse_args()

    source_image = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']
    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_image = resize(source_image, (256, 256))[..., :3]
    driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]
    generator, kp_detector, he_estimator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint, gen=opt.gen, cpu=opt.cpu)

    with open(opt.config) as f:
        config = yaml.load(f)
    estimate_jacobian = config['model_params']['common_params']['estimate_jacobian']
    print(f'estimate jacobian: {estimate_jacobian}')

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_image, driving_video, cpu=opt.cpu)
        print ("Best frame: " + str(i))
        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_image, driving_forward, generator, kp_detector, he_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        predictions_backward = make_animation(source_image, driving_backward, generator, kp_detector, he_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_image, driving_video, generator, kp_detector, he_estimator, relative=opt.relative, adapt_movement_scale=opt.adapt_scale, estimate_jacobian=estimate_jacobian, cpu=opt.cpu, free_view=opt.free_view, yaw=opt.yaw, pitch=opt.pitch, roll=opt.roll)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
