# DocXclassifier: towards a robust and interpretable deep neural network for document image classification
This repository contains the evaluation code for the paper [DocXclassifier: towards a robust and interpretable deep neural network for document image classification](https://link.springer.com/article/10.1007/s10032-024-00483-w) by Saifullah Saifullah, Stefan Agne, Andreas Dengel, and Sheraz Ahmed.

Requires Python 3+. For evaluation, please follow the steps below.

## Environment Setup
### Clone the repository
git clone https://github.com/saifullah3396/docxclassifier.git --recursive

### Install requirements
Install the dependencies:
```
pip install -r requirements.txt
```

### Setup environment variables:
```
export PYTHONPATH=./external/torchfusion/src
export DATA_ROOT_DIR=/home/ataraxia/Datasets/
export TORCH_FUSION_CACHE_DIR=</your/cache/dir>
export TORCH_FUSION_OUTPUT_DIR=</your/output/dir> # can be any directory where datasets are cached and model training outputs are generated.
```

### DocXClassifier Models
| Model | Dataset | Accuracy |
| :---: | :---: | :---: |
| [DocXClassifier-B](https://cloud.dfki.de/owncloud/index.php/s/spHfzK6Ezxc5MPb/download/RvlCdip_docxclassifier_base.pt) | RVL-CDIP | 94.00%
| [DocXClassifier-L](https://cloud.dfki.de/owncloud/index.php/s/bWNHCjobFo3YYAK/download/RvlCdip_docxclassifier_large.pt) | RVL-CDIP | 94.15%
| [DocXClassifier-XL](https://cloud.dfki.de/owncloud/index.php/s/ABBH35X7sJGXSyF/download/RvlCdip_docxclassifier_large_xlarge.pt) | RVL-CDIP | 94.17%
| [DocXClassifier-B](https://cloud.dfki.de/owncloud/index.php/s/2jqLQHB7kQsqZjx/download/Tobacco3482_rvlcdip_pretrained_docxclassifier_base.pt) | Tobacco3482 (RVL-CDIP Pretraining) | 95.29%
| [DocXClassifier-L](https://cloud.dfki.de/owncloud/index.php/s/dRZENH4QBbtQGtR/download/Tobacco3482_rvlcdip_pretrained_docxclassifier_large.pt) | Tobacco3482 (RVL-CDIP Pretraining) | 95.57%
| [DocXClassifier-XL](https://cloud.dfki.de/owncloud/index.php/s/KAbZANRnCoABpxb/download/Tobacco3482_rvlcdip_pretrained_docxclassifier_xlarge.pt) | Tobacco3482 (RVL-CDIP Pretraining) | 95.43%
| [DocXClassifier-B](https://cloud.dfki.de/owncloud/index.php/s/6xBA2qcD8mzAier/download/Tobacco3482_docxclassifier_base.pt) | Tobacco3482 (ImageNet Pretraining) | 87.43%
| [DocXClassifier-L](https://cloud.dfki.de/owncloud/index.php/s/yj93noqiMAijyqb/download/Tobacco3482_docxclassifier_large.pt) | Tobacco3482 (ImageNet Pretraining) | 88.43%
| [DocXClassifier-XL](https://cloud.dfki.de/owncloud/index.php/s/XWfjdz7nWMHEeo5/download/Tobacco3482_docxclassifier_xlarge.pt) | Tobacco3482 (ImageNet Pretraining) | 90.14%

### DocXClassifierFPN Models
| Model | Dataset | Accuracy |
| :---: | :---: | :---: |
| [DocXClassifierFPN-B](https://cloud.dfki.de/owncloud/index.php/s/5WAaiDpoZSBBbB7/download/RvlCdip_docxclassifier_fpn_base.pt) | RVL-CDIP | 94.04%
| [DocXClassifierFPN-L](https://cloud.dfki.de/owncloud/index.php/s/HWoPYQPsMeA5XXK/download/RvlCdip_docxclassifier_fpn_large.pt) | RVL-CDIP | 94.13%
| [DocXClassifierFPN-XL](https://cloud.dfki.de/owncloud/index.php/s/cRbpwQ3834HzMDQ/download/RvlCdip_docxclassifier_fpn_large_xlarge.pt) | RVL-CDIP | 94.19%
| [DocXClassifierFPN-B](https://cloud.dfki.de/owncloud/index.php/s/PGE2qiZb5waMKrx/download/Tobacco3482_rvlcdip_pretrained_docxclassifier_fpn_base.pt) | Tobacco3482 (RVL-CDIP Pretraining) | 95.57%
| [DocXClassifierFPN-L](https://cloud.dfki.de/owncloud/index.php/s/oKLWgdTAexGKEce/download/Tobacco3482_rvlcdip_pretrained_docxclassifier_fpn_large.pt) | Tobacco3482 (RVL-CDIP Pretraining) | 95.71%
| [DocXClassifierFPN-XL](https://cloud.dfki.de/owncloud/index.php/s/e4gQCiWK34aP6Wc/download/Tobacco3482_rvlcdip_pretrained_docxclassifier_fpn_xlarge.pt) | Tobacco3482 (RVL-CDIP Pretraining) | 94.86%
| [DocXClassifierFPN-B](https://cloud.dfki.de/owncloud/index.php/s/JB2CyAwGqYxdG5W/download/Tobacco3482_docxclassifier_fpn_base.pt) | Tobacco3482 (ImageNet Pretraining) | 88.43%
| [DocXClassifierFPN-L](https://cloud.dfki.de/owncloud/index.php/s/HjHCFoYkkbaPR7o/download/Tobacco3482_docxclassifier_fpn_large.pt) | Tobacco3482 (ImageNet Pretraining) | 89.57%
| [DocXClassifierFPN-XL](https://cloud.dfki.de/owncloud/index.php/s/R3y8e5HNrAXPMto/download/Tobacco3482_docxclassifier_fpn_xlarge.pt) | Tobacco3482 (ImageNet Pretraining) | 90.29%

# Evaluation on RVL-CDIP:
Please download the [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset and place it under the directory $DATA_ROOT_DIR/documents/rvlcdip.
Evaluate the DocXClassifier models on the RVL-CDIP dataset using the following script:
```
./scripts/run/evaluate/document_classification/evaluate_rvlcdip_no_fpn.sh
```

Evaluate the DocXClassifierFPN models on the RVL-CDIP dataset using the following script:
```
./scripts/run/evaluate/document_classification/evaluate_rvlcdip_fpn.sh
```

# Evaluation on Tobacco3482 dataset:
Please download the [Tobacco3482](https://www.kaggle.com/patrickaudriaz/tobacco3482jpg) dataset and place it under the directory $DATA_ROOT_DIR/documents/tobacco3482.
Evaluate the DocXClassifier models on the Tobacco3482 dataset with ImageNet pretraining using the following script:
```
./scripts/run/evaluate/document_classification/evaluate_tobacco3482_no_fpn.sh
```

Evaluate the DocXClassifier models on the Tobacco3482 dataset with ImageNet pretraining using the following script:
```
./scripts/run/evaluate/document_classification/evaluate_tobacco3482_fpn.sh
```

Evaluate the DocXClassifier models on the Tobacco3482 dataset with RVL-CDIP pretraining using the following script:
```
./scripts/run/evaluate/document_classification/evaluate_tobacco3482_rvlcdip_pretrained_no_fpn.sh
```

Evaluate the DocXClassifier models on the Tobacco3482 dataset with RVL-CDIP pretraining using the following script:
```
./scripts/run/evaluate/document_classification/evaluate_tobacco3482_rvlcdip_pretrained_fpn.sh
```

# Citation
If you find this useful in your research, please consider citing our associated paper:
```
@article{Saifullah2024,
  title = {DocXclassifier: towards a robust and interpretable deep neural network for document image classification},
  ISSN = {1433-2825},
  url = {http://dx.doi.org/10.1007/s10032-024-00483-w},
  DOI = {10.1007/s10032-024-00483-w},
  journal = {International Journal on Document Analysis and Recognition (IJDAR)},
  publisher = {Springer Science and Business Media LLC},
  author = {Saifullah,  Saifullah and Agne,  Stefan and Dengel,  Andreas and Ahmed,  Sheraz},
  year = {2024},
  month = jun
}
```

# License
This repository is released under the Apache 2.0 license as found in the LICENSE file.


