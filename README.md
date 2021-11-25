# One-Shot Free-View Neural Talking Head Synthesis
Unofficial pytorch implementation of paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing".  

```Python 3.6``` and ```Pytorch 1.7``` are used. 


Updates:  
-------- 
```2021.11.05``` :
* <s>Replace Jacobian with the rotation matrix (Assuming J = R) to avoid estimating Jacobian.</s> (not working)
* Correct the rotation matrix.

```2021.11.17``` :
* Better Generator, better performance (models and checkpoints have been released).  

Driving | Beta Version | FOMM | New Version:  


https://user-images.githubusercontent.com/17874285/142828000-db7b324e-c2fd-4fdc-a272-04fb8adbc88a.mp4


--------
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
python demo.py --config config/vox-256.yaml --checkpoint path/to/checkpoint --source_image path/to/source --driving_video path/to/driving --relative --adapt_scale --find_best_frame --free_view --yaw 20 --pitch 0 --roll 0
```
Note: run ```crop-video.py --inp driving_video.mp4``` first to get the cropping suggestion and crop the raw video.  

Pretrained Model:  
--------

  Model  |  Train Set   | Baidu Netdisk | Media Fire | 
 ------- |------------  |-----------    |--------      |
 Vox-256-Beta| VoxCeleb-v1  | [Baidu](https://pan.baidu.com/s/1lLS4ArbK2yWelsL-EtwU8g) (PW: c0tc)|  [MF](https://www.mediafire.com/folder/rw51an7tk7bh2/TalkingHead)  |
 Vox-256-New | VoxCeleb-v1  |  -  |  [MF](https://www.mediafire.com/folder/fcvtkn21j57bb/TalkingHead_Update)  |
 Vox-512 | VoxCeleb-v2  |  soon  |  soon  |
 
 Note:
 1. <s>For now, the Beta Version is not well tuned.</s>
 2. For free-view synthesis, it is recommended that Yaw, Pitch and Roll are within ±45°, ±20° and ±20° respectively.
 3. Face Restoration algorithms ([GPEN](https://github.com/yangxy/GPEN)) can be used for post-processing to significantly improve the resolution.
![show](https://github.com/zhanglonghao1992/ReadmeImages/blob/master/images/s%20r.gif) 


Acknowlegement: 
--------
Thanks to [NV](https://github.com/NVlabs/face-vid2vid), [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model) and [DeepHeadPose](https://github.com/DriverDistraction/DeepHeadPose).
