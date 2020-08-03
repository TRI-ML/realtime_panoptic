## Real-Time Panoptic Segmentation from Dense Detections

Official [PyTorch](https://pytorch.org/) implementation of the CVPR 2020 Oral **Real-Time Panoptic Segmentation from Dense Detections** by the ML Team at [Toyota Research Institute (TRI)](https://www.tri.global/), cf. [References](#references) below.

<a href="https://www.tri.global/" target="_blank">
 <img align="right" src="/media/figs/tri-logo.png" width="20%"/>
</a>

<a href="https://www.youtube.com/watch?v=xrxaRU2g2vo" target="_blank">
<img width="60%" src="/media/figs/panoptic-teaser.gif"/>
</a>

## Install
```
git clone https://github.com/TRI-ML/realtime_panoptic.git
cd realtime_panoptic
make docker-build
```

To verify your installation, you can also run our simple test run to conduct inference on 1 test image using our Cityscapes pretrained model:
```
make docker-run-test-sample
```

Now you can start a docker container with interactive mode:
```
make docker-start
```
## Demo
We provide demo code to conduct inference on Cityscapes pretrained model. 
```
python scripts/demo.py --config-file <config.yaml>  --input <input_image_file> \
        --pretrained-weight <checkpoint.pth>
```
Simple user example using our pretrained model previded in the Models section:
```
python scripts/demo.py --config-file ./configs/demo_config.yaml --input media/figs/test.png --pretrained-weight cvpr_realtime_pano_cityscapes_standalone_no_prefix.pth
```

## Models
 

### Cityscapes
| Model |  PQ | PQ_th | PQ_st | 
| :--- | :---: | :---: | :---: | 
| [ResNet-50](https://tri-ml-public.s3.amazonaws.com/github/realtime_panoptic/models/cvpr_realtime_pano_cityscapes_standalone_no_prefix.pth) | 58.8 | 52.1| 63.7 |

## License

The source code is released under the [MIT license](LICENSE.md).

## References

#### Real-Time Panoptic Segmentation from Dense Detections (CVPR 2020 oral)
*Rui Hou\*, Jie Li\*, Arjun Bhargava, Allan Raventos, Vitor Guizilini, Chao Fang, Jerome Lynch, Adrien Gaidon*, [**[paper]**](https://arxiv.org/abs/1912.01202), [**[oral presentation]**](https://www.youtube.com/watch?v=xrxaRU2g2vo), [**[teaser]**](https://www.youtube.com/watch?v=_N4kGJEg-rM)
```
@InProceedings{real-time-panoptic,
author = {Hou, Rui and Li, Jie and Bhargava, Arjun and Raventos, Allan and Guizilini, Vitor and Fang, Chao and Lynch, Jerome and Gaidon, Adrien},
title = {Real-Time Panoptic Segmentation From Dense Detections},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
