# Pixel-Level Cycle Association
This is the Pytorch implementation of our NeurIPS 2020 Oral paper [Pixel-Level Cycle Association: A New Perspective for Domain Adaptive Semantic Segmentation](https://proceedings.neurips.cc/paper/2020/file/243be2818a23c980ad664f30f48e5d19-Paper.pdf). 

## Requirements
```
pip install -r ./requirements.txt
```
We test our codes with two NVIDIA Tesla V100 (32G) GPU cards.

## Dataset
See ```experiments/data/```

## Pre-trained Model
Following general practice, our training starts from [ResNet-101](https://drive.google.com/file/d/11ULk5WkPVMUmuEs8nmMJVgm5gkt9ZMfN/view?usp=sharing) backbone pretrained on ImageNet. Please download the weight file and put it under the ```model``` directory. 

## Training
For GTAV to CityScapes:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env ./tools/train.py --cfg ./experiment/config/g2c_train.yaml --exp_name g2c 
```
For SYNTHIA to CityScapes:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env ./tools/train.py --cfg ./experiment/config/s2c_train.yaml --exp_name s2c 
```

You can also use the shell scripts provided under directory ```experiment/scripts/train.sh``` to train your model.

## Test

For GTAV to CityScapes:
```
CUDA_VISIBLE_DEVICES=0,1 python ./tools/test.py --cfg ./experiment/config/g2c_test.yaml --weights ${trained_weights} --exp_name g2c_test
```
For SYNTHIA to CityScapes:
```
CUDA_VISIBLE_DEVICES=0,1 python ./tools/test.py --cfg ./experiment/config/s2c_test.yaml --weights ${trained_weights} --exp_name s2c_test
```

You can also use the shell scripts provided under directory ```experiment/scripts/test_normal.sh``` to evaluate your model.


## Citing 
Please cite our paper if you use our code in your research:
```
@inproceedings{kang2020pixel,
  title={Pixel-Level Cycle Association: A New Perspective for Domain Adaptive Semantic Segmentation},
  author={Kang, Guoliang and Wei, Yunchao and Yang, Yi and Zhuang, Yueting and Hauptmann, Alexander G},
  booktitle={NeurIPS},
  year={2020}
}
```
## Contact
If you have any questions, please contact me via kgl.prml@gmail.com.

## Thanks to third party
torchvision 

[LovaszSoftmax](https://github.com/bermanmaxim/LovaszSoftmax)

