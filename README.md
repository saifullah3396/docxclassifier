# DocXClassifier: High Performance Explainable Deep Network for Document Image Classification
This repository contains the evaluation code for the paper [DocXClassifier: High Performance Explainable Deep Network for Document Image Classification](https://www.techrxiv.org/articles/preprint/DocXClassifier_High_Performance_Explainable_Deep_Network_for_Document_Image_Classification/19310489) by Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please follow the steps below.

# Requirements
Please install the requirements with pip as follows:
```
pip install -r requirements.txt
```

Set PYTHONPATH to match source the directory:
```
export PYTHONPATH=`pwd`/src
```

Create output directory for holding dataset, models, etc
```
export OUTPUT=</path/to/output/>
mkdir -p $OUTPUT
```

# Models
| Model | Dataset | Accuracy |
| :---: | :---: | :---: |
| [DocXClassifier-B](https://cloud.dfki.de/owncloud/index.php/s/5e5bjtNe56yLYTy/download/base_rvlcdip.pth) | RVL-CDIP | 93.74
| [DocXClassifier-L](https://cloud.dfki.de/owncloud/index.php/s/yoPzK6T9RHX4C7C/download/large_rvlcdip.pth) | RVL-CDIP | 93.75
| [DocXClassifier-XL](https://cloud.dfki.de/owncloud/index.php/s/X9HXS7HJT5FRBN2/download/xlarge_rvlcdip.pth) | RVL-CDIP |  94.07
| [DocXClassifier-B](https://cloud.dfki.de/owncloud/index.php/s/QybGNHkAXKqDypD/download/base_tobacco_rvlcdip.pth) | Tobacco3482 (RVL-CDIP Pretraining) | 94.71
| [DocXClassifier-L](https://cloud.dfki.de/owncloud/index.php/s/TxZAR9Mo6bFiKWy/download/large_tobacco_rvlcdip.pth) | Tobacco3482 (RVL-CDIP Pretraining) | 94.86
| [DocXClassifier-XL](https://cloud.dfki.de/owncloud/index.php/s/b8J3Xf8K5EDKLc9/download/xlarge_tobacco_rvlcdip.pth) | Tobacco3482 (RVL-CDIP Pretraining) | 95.29
| [DocXClassifier-B](https://cloud.dfki.de/owncloud/index.php/s/m2XR3yL3TFKesCx/download/base_tobacco_imagenet.pth) | Tobacco3482 (ImageNet Pretraining) | 87.43
| [DocXClassifier-L](https://cloud.dfki.de/owncloud/index.php/s/r88txKxbnx3s46N/download/large_tobacco_imagenet.pth) | Tobacco3482 (ImageNet Pretraining) | 88.43
| [DocXClassifier-XL](https://cloud.dfki.de/owncloud/index.php/s/TEfnWQ89ZbHBnG3/download/xlarge_tobacco_imagenet.pth) | Tobacco3482 (ImageNet Pretraining) | 90.14

# Evaluation on RVL-CDIP:
Please download the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
Evaluate the models as follows:
```
./scripts/evaluate.sh --cfg ./cfg/rvlcdip/base.yaml data_args.dataset_dir </path/to/rvlcdip_dataset> # base model
```
```
./scripts/evaluate.sh --cfg ./cfg/rvlcdip/large.yaml data_args.dataset_dir </path/to/rvlcdip_dataset> # large model
```
```
./scripts/evaluate.sh --cfg ./cfg/rvlcdip/xlarge.yaml data_args.dataset_dir </path/to/rvlcdip_dataset> # xlarge model
```


# Evaluation on Tobacco3482 dataset:
Please download the [Tobacco3482](https://www.kaggle.com/patrickaudriaz/tobacco3482jpg) dataset and put it in the /path/to/tobacco3482 directory. Make sure to keep train.txt and test.txt files from the data/tobacco3482 directory.

Copy the train/test splits to the dataset directory:
```
cp data/tobacco3482/* /path/to/tobacco3482/
```

## ImageNet Pretraining
Evaluate the models as follows:
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_imagenet/base.yaml data_args.dataset_dir </path/to/tobacco3482_dataset> # base model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_imagenet/large.yaml data_args.dataset_dir </path/to/tobacco3482_dataset> # large model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_imagenet/xlarge.yaml data_args.dataset_dir </path/to/tobacco3482_dataset> # xlarge model
```

## RVL-CDIP Pretraining
Evaluate the models as follows:
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_rvlcdip/base.yaml data_args.dataset_dir </path/to/tobacco3482_dataset># base model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_rvlcdip/large.yaml data_args.dataset_dir </path/to/tobacco3482_dataset># large model
```
```
./scripts/evaluate.sh --cfg ./cfg/tobacco3482_rvlcdip/xlarge.yaml data_args.dataset_dir </path/to/tobacco3482_dataset># xlarge model
```

# Generating Attention Maps
To generate the attention maps just call the following script:
```
./scripts/generate_attention_maps.sh --cfg ./cfg/rvlcdip/base.yaml data_args.dataset_dir </path/to/rvlcdip_dataset> # base model
```
This will save the attention maps from the model to the attn directory. For different models just change the --cfg as above.


# Citation
If you find this useful in your research, please consider citing our associated paper:
```
@misc{
  docxclassifier-saifullah2022, 
  title={DocXClassifier: High Performance Explainable Deep Network for Document Image Classification}, 
  url={https://www.techrxiv.org/articles/preprint/DocXClassifier_High_Performance_Explainable_Deep_Network_for_Document_Image_Classification/19310489/2}, 
  DOI={10.36227/techrxiv.19310489.v2},
  publisher={TechRxiv}, 
  author={Saifullah and Agne, Stefan and Dengel, Andreas and Ahmed, Sheraz}, 
  year={2022}, 
  month={Mar} 
} 
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.
