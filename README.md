# CPFNet_Project

## Introduction
This repository is for the PAPER: ***CPFNet: Context Pyramid Fusion Network for Medical Image Segmentation***, which has been accepted by IEEE TRANSACTIONS ON MEDICAL IMAGING 


The author list： Shuanglang Feng, Heming Zhao, Fei Shi, Xuena Cheng, Meng Wang, Yuhui Ma, Dehui Xiang, Weifang Zhu and Xinjian Chen from SooChow University.

**The source code is now available!**
## citation

```
@inproceedings{feng2020cpfnet,
    author={Shuanglang Feng and Heming Zhao and Fei Shi and Xuena Cheng and Meng Wang and Yuhui Ma and Dehui Xiang and Weifang Zhu and Xinjian Chen},
    title={CPFNet: Context Pyramid Fusion Network for Medical Image Segmentation},
    booktitle={IEEE TRANSACTIONS ON MEDICAL IMAGING},   
    year={2020},   
}
```
## Folder
- `Dataset`: the folder where dataset is placed.
- `OCT`: the folder where model and model environment code are placed,`OCT` is the name of task. 
   - `dataset`: the file of data preprocessing.
   - `model`: model files.
   - `utils`: utils files(include many utils)
      - `config.py`: some configuration about project parameters.
      - `loss.py`: some custom loss functions
      - `utils.py`: some definitions of evaluation indicators
   - `metric.py`: offline evaluation function
   - `train.py`: training, validation and test function. 
- `Pretrain_model`:  pretriand encoder model,for example,resnet34.

## Prerequisites
- PyTorch 1.0   
   - `conda install torch torchvision`
- tqdm
   - `conda install tqdm`
- imgaug
   - `conda install six numpy scipy Pillow matplotlib scikit-image opencv-python imageio Shapely`
   - `conda install imgaug`
## 
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
![image](https://user-images.githubusercontent.com/38462831/129555294-0a2bf836-084d-462a-ba03-41003b434d4d.png)
