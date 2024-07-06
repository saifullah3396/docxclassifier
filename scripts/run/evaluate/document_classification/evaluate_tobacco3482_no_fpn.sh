#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# first evaluate models without the FPN. The original models were trained using CV2 resizing instead of PIL resizing. This causes huge differences in perfs so we keep it the same
BASE_PATH="https://cloud.dfki.de/owncloud/index.php/s/6xBA2qcD8mzAier/download"
CHECKPOINT_PATH="${BASE_PATH}/Tobacco3482_docxclassifier_base.pt"
$SCRIPT_DIR/../../../../scripts/evaluate.sh \
    +evaluate/document_classification=with_aug_and_preprocess \
    base/data_args=document_classification/tobacco3482 \
    dataset_config_name=default \
    base/model_args=docxclassifier/convnext_base \
    image_size_y=384 \
    image_size_x=384 \
    preprocess_image_size_x=384 \
    preprocess_image_size_y=384 \
    per_device_train_batch_size=64 \
    per_device_eval_batch_size=64 \
    dataloader_num_workers=4 \
    experiment_name=evaluate_docxclassifier_b \
    checkpoint=$CHECKPOINT_PATH \
    checkpoint_state_dict_key=model \
    resize_pil_or_cv2=CV2 \
    +use_fpn=False \
    $@

# first evaluate models without the FPN. The original models were trained using CV2 resizing instead of PIL resizing. This causes huge differences in perfs so we keep it the same
BASE_PATH="https://cloud.dfki.de/owncloud/index.php/s/yj93noqiMAijyqb/download"
CHECKPOINT_PATH="${BASE_PATH}/Tobacco3482_docxclassifier_large.pt"
$SCRIPT_DIR/../../../../scripts/evaluate.sh \
    +evaluate/document_classification=with_aug_and_preprocess \
    base/data_args=document_classification/tobacco3482 \
    dataset_config_name=default \
    base/model_args=docxclassifier/convnext_large \
    image_size_y=384 \
    image_size_x=384 \
    preprocess_image_size_x=384 \
    preprocess_image_size_y=384 \
    per_device_train_batch_size=64 \
    per_device_eval_batch_size=64 \
    dataloader_num_workers=4 \
    experiment_name=evaluate_docxclassifier_l \
    checkpoint=$CHECKPOINT_PATH \
    checkpoint_state_dict_key=model \
    resize_pil_or_cv2=CV2 \
    +use_fpn=False \
    $@

# first evaluate models without the FPN. The original models were trained using CV2 resizing instead of PIL resizing. This causes huge differences in perfs so we keep it the same
BASE_PATH="https://cloud.dfki.de/owncloud/index.php/s/XWfjdz7nWMHEeo5/download"
CHECKPOINT_PATH="${BASE_PATH}/Tobacco3482_docxclassifier_xlarge.pt"
$SCRIPT_DIR/../../../../scripts/evaluate.sh \
    +evaluate/document_classification=with_aug_and_preprocess \
    base/data_args=document_classification/tobacco3482 \
    dataset_config_name=default \
    base/model_args=docxclassifier/convnext_xlarge \
    image_size_y=384 \
    image_size_x=384 \
    preprocess_image_size_x=384 \
    preprocess_image_size_y=384 \
    per_device_train_batch_size=64 \
    per_device_eval_batch_size=64 \
    dataloader_num_workers=4 \
    experiment_name=evaluate_docxclassifier_xl \
    checkpoint=$CHECKPOINT_PATH \
    checkpoint_state_dict_key=model \
    resize_pil_or_cv2=CV2 \
    +use_fpn=False \
    $@
