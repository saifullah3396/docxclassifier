#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# first evaluate models without the FPN. The original models were trained using CV2 resizing instead of PIL resizing. This causes huge differences in perfs so we keep it the same
BASE_PATH="https://cloud.dfki.de/owncloud/index.php/s/PGE2qiZb5waMKrx/download"
CHECKPOINT_PATH="${BASE_PATH}/Tobacco3482_rvlcdip_pretrained_docxclassifier_fpn_base.pt"
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
    resize_pil_or_cv2=PIL \
    +use_fpn=True \
    $@

# first evaluate models without the FPN. The original models were trained using CV2 resizing instead of PIL resizing. This causes huge differences in perfs so we keep it the same
BASE_PATH="https://cloud.dfki.de/owncloud/index.php/s/oKLWgdTAexGKEce/download"
CHECKPOINT_PATH="${BASE_PATH}/Tobacco3482_rvlcdip_pretrained_docxclassifier_fpn_large.pt"
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
    resize_pil_or_cv2=PIL \
    +use_fpn=True \
    $@

# first evaluate models without the FPN. The original models were trained using CV2 resizing instead of PIL resizing. This causes huge differences in perfs so we keep it the same
BASE_PATH="https://cloud.dfki.de/owncloud/index.php/s/e4gQCiWK34aP6Wc/download"
CHECKPOINT_PATH="${BASE_PATH}/Tobacco3482_rvlcdip_pretrained_docxclassifier_fpn_xlarge.pt"
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
    resize_pil_or_cv2=PIL \
    +use_fpn=True \
    $@
