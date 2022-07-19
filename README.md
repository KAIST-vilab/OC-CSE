
# Unlocking the Potential of Ordinary Classifier: Class-specific Adversarial Erasing Framework for Weakly Supervised Semantic Segmentation

This repository contains the official PyTorch implementation of the paper "[Unlocking the Potential of Ordinary Classifier: Class-specific Adversarial Erasing Framework for Weakly Supervised Semantic Segmentation](https://openaccess.thecvf.com/content/ICCV2021/papers/Kweon_Unlocking_the_Potential_of_Ordinary_Classifier_Class-Specific_Adversarial_Erasing_Framework_ICCV_2021_paper.pdf)" paper (ICCV 2021) by [Hyeokjun Kweon](https://github.com/sangrockEG) and [Sung-Hoon Yoon](https://github.com/sunghoonYoon).

<img src = "https://user-images.githubusercontent.com/42232407/128456385-a596a274-5803-44b4-8720-3830aad753de.PNG" width="60%"><img src = "https://user-images.githubusercontent.com/42232407/128457060-4777b7d3-0ec8-4b61-8ea5-e9149fd98de8.png" width="40%">

## Introduction
We have developed a framework that extract the potential of the ordinary classifier with class-specific adversarial erasing framework for weakly supervised semantic segmentation.
With image-level supervision only, we achieved new state-of-the-arts both on PASCAL VOC 2012 and MS-COCO.

## Citation
If our code be useful for you, please consider citing our ICCV paper using the following BibTeX entry.
```
@inproceedings{kweon2021unlocking,
  title={Unlocking the potential of ordinary classifier: Class-specific adversarial erasing framework for weakly supervised semantic segmentation},
  author={Kweon, Hyeokjun and Yoon, Sung-Hoon and Kim, Hyeonseong and Park, Daehee and Yoon, Kuk-Jin},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={6994--7003},
  year={2021}
}
```
## Prerequisite
* Tested on Ubuntu 16.04, with Python 3.6, PyTorch 1.5.1, CUDA 10.1, both on both single and multi gpu.
* You can create conda environment with the provided yaml file.
```
conda env create -f od_cse.yaml
```
* [The PASCAL VOC 2012 development kit](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/):
You need to specify place VOC2012 under ./data folder.
* ImageNet-pretrained weights for resnet38d are from [[resnet_38d.params]](https://github.com/itijyou/ademxapp).
You need to place the weights as ./pretrained/resnet_38d.params.
* PASCAL-pretrained weights for resnet38d are from [[od_cam.pth]](https://github.com/jiwoon-ahn/psa).
You need to place the weights as ./pretrained/od_cam.pth.
## Usage
### Training
* Please specify the name of your experiment.
* Training results are saved at ./experiment/[exp_name]
```
python train.py --name [exp_name] --model model_cse
```
### Inference
```
python infer.py --name [exp_name] --model model_cse --load_epo [epoch_to_load] --vis --dict --crf --alphas 6 10 24
```
### Evaluation for CAM result
```
python evaluation.py --name [exp_name] --task cam --dict_dir dict
```
### Evaluation for CRF result (ex. alpha=6)
```
python evaluation.py --name [exp_name] --task crf --dict_dir crf/06
```

we heavily borrow the work from [AffinityNet](https://github.com/jiwoon-ahn/psa) repository. Thanks for the excellent codes!
