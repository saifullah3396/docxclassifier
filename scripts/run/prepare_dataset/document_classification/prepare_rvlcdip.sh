#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# $SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
#     +prepare_dataset/document_classification=with_aug_and_preprocess \
#     base/data_args=document_classification/rvlcdip \
#     dataset_config_name=default_test \
#     image_size_y=224 \
#     image_size_x=224 \
#     preprocess_image_size_x=224 \
#     preprocess_image_size_y=224 \
#     resize_pil_or_cv2=CV2 \
#     $@

# $SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
#     +prepare_dataset/document_classification=with_aug_and_preprocess \
#     base/data_args=document_classification/rvlcdip \
#     dataset_config_name=default_test \
#     image_size_y=224 \
#     image_size_x=224 \
#     preprocess_image_size_x=224 \
#     preprocess_image_size_y=224 \
#     resize_pil_or_cv2=PIL \
#     $@

$SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
    +prepare_dataset/document_classification=with_aug_and_preprocess \
    base/data_args=document_classification/rvlcdip \
    dataset_config_name=default_test \
    image_size_y=384 \
    image_size_x=384 \
    preprocess_image_size_x=384 \
    preprocess_image_size_y=384 \
    resize_pil_or_cv2=CV2 \
    $@

$SCRIPT_DIR/../../../../scripts/prepare_dataset.sh \
    +prepare_dataset/document_classification=with_aug_and_preprocess \
    base/data_args=document_classification/rvlcdip \
    dataset_config_name=default_test \
    image_size_y=384 \
    image_size_x=384 \
    preprocess_image_size_x=384 \
    preprocess_image_size_y=384 \
    resize_pil_or_cv2=PIL \
    $@
