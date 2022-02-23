# DocXClassifier: High Performance Explainable Deep Network for Document Image Classification
This repository contains the evaluation code for the paper [DocXClassifier: High Performance Explainable Deep Network for Document Image Classification](https) by Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please follow the steps below.

# Requirements
Please install the requirements with pip as follows:
```
pip install -r requirements.txt
```

# Evaluation on RVL-CDIP:
Please download the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset and put it in the data directory.
Evaluate the models as follows:
```
./scripts/evaluate.sh --cfg ./cfg/rvlcdip/base.yaml # base model
```
```
./scripts/evaluate.sh --cfg ./cfg/rvlcdip/large.yaml # large model
```
```
./scripts/evaluate.sh --cfg ./cfg/rvlcdip/xlarge.yaml # xlarge model
```


# Evaluation on Tobacco3482 dataset:
Please download the [Tobacco3482](https://www.kaggle.com/patrickaudriaz/tobacco3482jpg) dataset and put it in the data/tobacco3482 directory. Make sure to keep train.txt and test.txt files from the data/tobacco3482 directory.

## ImageNet Pretraining
Evaluate the models as follows:
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_imagenet/base.yaml # base model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_imagenet/large.yaml # large model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_imagenet/xlarge.yaml # xlarge model
```

## RVL-CDIP Pretraining
Evaluate the models as follows:
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_rvlcdip/base.yaml # base model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_rvlcdip/large.yaml # large model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_rvlcdip/xlarge.yaml # xlarge model
```

# Citation
If you find this useful in your research, please consider citing:
```
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.