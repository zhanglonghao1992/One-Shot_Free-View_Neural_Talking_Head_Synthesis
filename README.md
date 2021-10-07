# One-Shot Free-View Neural Talking Head Synthesis
Unofficial pytorch implementation of paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing".  

I‘ve only tried on ```python 3.6``` and ```pytorch 1.7.0```. 

Driving | FOMM | Ours:    
![show](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/081.gif) 

Free-View:  
![show](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/concat.gif) 

Train:  
--------
```
python run.py --config config/vox-256.yaml --device_ids 0,1,2,3,4,5,6,7
```

Demo:  
--------
```
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame
```
free-view (e.g. yaw=20, pitch=roll=0):
```
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame --free_view --yaw 20 --pitch 0 --row 0
```
Note: run ```crop-video.py --inp driving_video.mp4``` first to get the cropping suggestion and crop the raw video.  

Pretrained Model:  
--------

  Model  |  Train Set   | Baidu Netdisk | Meida Fire | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | [Baidu](https://pan.baidu.com/s/1lLS4ArbK2yWelsL-EtwU8g) (PW: c0tc)|  soon  |
 Vox-256-Stable| VoxCeleb-v1  |  soon  |  soon  |
 Vox-256 | VoxCeleb-v2  |  soon  |  soon  |
 Vox-512 | VoxCeleb-v2  |  soon  |  soon  |
 
 Note:
 1. At present, the Beta Version is not well tuned, the definition of synthesized image is poor, and the mouth shape and eyes are not very accurate.
 2. For free-view synthesis, it is recommended that Yaw, Pitch and Roll are within ±45°, ±20° and ±20° respectively.

Acknowlegement: 
--------
Thanks to [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model) and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose).
