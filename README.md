# One-Shot Free-View Neural Talking Head Synthesis
Unofficial pytorch implementation of paper "One-Shot Free-View Neural Talking-Head Synthesis for Video Conferencing"  


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
Note: run ```crop-video.py --inp driving_video.mp4``` to get crop suggestions and crop the driving video as prompted.

Acknowlegement: 
--------
Thanks to [face-vid2vid](https://github.com/NVlabs/face-vid2vid), and [AliaksandrSiarohin](https://github.com/AliaksandrSiarohin/first-order-model)
