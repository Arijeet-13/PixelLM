import glob
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import (ANSWER_LIST, DEFAULT_IMAGE_TOKEN, LONG_QUESTION_LIST)
NO_TARGET_ANSWER = "There is no target object in the image."


def transform_mask(masks, size):
    height, width = masks.shape[-2:]
    short, long = (width, height) if width <= height else (height, width)
    new_short, new_long = size, int(size * long / short)
    new_shape = (new_long, new_short) if width <= height else (new_short, new_long)
    masks = F.interpolate(masks[None].float(), size=new_shape, mode="nearest")[0].bool()

    orig_height, orig_width = new_shape
    crop_height, crop_width = int(size), int(size)
    top = (orig_height - crop_height) // 2
    bottom = top + crop_height
    left = (orig_width - crop_width) // 2
    right = left + crop_width
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width
    masks = masks[..., top:bottom, left:right]
    return masks

def _find_liss4reason_triplets(base_image_dir, ds_name, split,
                                images_dirname, labels_dirname, qas_dirname):
    root = os.path.join(base_image_dir, ds_name, split)
    images_root = os.path.join(root, images_dirname)
    labels_root = os.path.join(root, labels_dirname)
    qas_root = os.path.join(root, qas_dirname)

    images, labels, qas = [], [], []
    for qa_path in sorted(glob.glob(os.path.join(qas_root, "*.json"))):
        sample_id = os.path.splitext(os.path.basename(qa_path))[0]
        img_candidates = glob.glob(os.path.join(images_root, sample_id + ".*"))
        label_candidates = glob.glob(os.path.join(labels_root, sample_id + ".*"))
        if not img_candidates or not label_candidates:
            continue  # skip incomplete triplets rather than crashing mid-run
        images.append(img_candidates[0])
        labels.append(label_candidates[0])
        qas.append(qa_path)
    return images, labels, qas


class LISS4ReasonDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        precision: str = "fp32",
        image_size: int = 224,
        num_classes_per_sample: int = 1,   # LISS4Reason is one-target-per-image
        exclude_val=False,
        liss4_reason_data="LISS4Reason|train",
        num_classes_per_question=1,
        seg_token_num=1,
        pad_train_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        use_expand_question_list=False,
        images_dirname="images",
        labels_dirname="labels",
        qas_dirname="qa",
    ):
        self.exclude_val = exclude_val
        self.samples_per_epoch = samples_per_epoch
        self.num_classes_per_sample = num_classes_per_sample

        self.base_image_dir = base_image_dir
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.precision = precision
        self.transform = ResizeLongestSide(image_size)

        self.long_question_list = list(LONG_QUESTION_LIST)
        self.answer_list = ANSWER_LIST
        self.seg_token_num = seg_token_num
        self.num_classes_per_question = num_classes_per_question

        self.masks_process_with_clip = masks_process_with_clip
        self.pad_train_clip_images = pad_train_clip_images
        self.clip_image_processor = (
            CLIPImageProcessor.from_pretrained(vision_tower)
            if preprocessor_config == ''
            else CLIPImageProcessor.from_pretrained(preprocessor_config)
        )
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])

        dataset_name, split = liss4_reason_data.split("|")
        splits = split.split("_")

        images, labels, qas = [], [], []
        for sp in splits:
            sp_images, sp_labels, sp_qas = _find_liss4reason_triplets(
                base_image_dir, dataset_name, sp, images_dirname, labels_dirname, qas_dirname
            )
            images.extend(sp_images)
            labels.extend(sp_labels)
            qas.extend(sp_qas)

        self.images = images
        self.labels = labels
        self.qas = qas

        print("number of LISS4Reason samples: ", len(images))

    def __len__(self):
        return self.samples_per_epoch

    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.images) - 1)
        image_path = self.images[idx]
        label_path = self.labels[idx]
        qa_path = self.qas[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_size = image.shape[:2]

        # ---- CLIP preprocessing ----
        if self.pad_train_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(
                torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(),
                self.clip_image_processor.size['shortest_edge'],
            )
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        raw_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"Could not read mask: {label_path}")
        binary_mask = (raw_mask != 0).astype(np.float32)

        with open(qa_path, "r") as f:
            qa = json.load(f)
        questions_pool = qa.get("questions", [])
        answers_pool = qa.get("answer", [])
        has_target = len(answers_pool) > 0 and binary_mask.sum() > 0

        question_text = questions_pool[0].strip() if questions_pool else \
            "What is the target region in this image?"

        image = self.transform.apply_image(image)  # preprocess image for sam
        resize = image.shape[:2]

        seg_token = ["[SEG{}]".format(i) for i in range(self.seg_token_num)]
        seg_token = ' '.join(seg_token)

        question_template = self.long_question_list[0]
        question = question_template.format(sent=question_text)

        if has_target:
            free_text_answer = answers_pool[0].strip()
            seg_answer_template = self.answer_list[0] if self.seg_token_num == 1 \
                else self.answer_list[0].replace('[SEG]', seg_token)
            answer = seg_answer_template + " {}".format(free_text_answer)
        else:
            answer = NO_TARGET_ANSWER

        questions = [question]
        answers = [answer]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], questions[0])
        conv.append_message(conv.roles[1], answers[0])
        conversations.append(conv.get_prompt())

        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        if has_target:
            masks = np.stack([binary_mask], axis=0)
            masks = torch.from_numpy(masks)
            label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        else:
            # no mask to supervise on for this sample
            masks = torch.rand(0, *ori_size)
            label = torch.ones(ori_size) * self.ignore_label

        sampled_sents = [question_text]

        if self.masks_process_with_clip:
            mask_shape = image_clip.shape[-1]
            if len(masks) == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                masks = transform_mask(masks, mask_shape)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            label,
            resize,
            clip_resize,
            questions,
            sampled_sents,
        )


class LISS4ReasonValDataset(torch.utils.data.Dataset):

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        val_dataset,  # e.g. "LISS4Reason|val"
        image_size=1024,
        seg_token_num=1,
        pad_val_clip_images=False,
        masks_process_with_clip=False,
        preprocessor_config='',
        images_dirname="images",
        labels_dirname="labels",
        qas_dirname="qa",
    ):
        self.seg_token_num = seg_token_num
        self.base_image_dir = base_image_dir
        self.pad_val_clip_images = pad_val_clip_images
        self.masks_process_with_clip = masks_process_with_clip
        self.image_size = image_size
        self.tokenizer = tokenizer
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = (
            CLIPImageProcessor.from_pretrained(vision_tower)
            if preprocessor_config == ''
            else CLIPImageProcessor.from_pretrained(preprocessor_config)
        )
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])

        ds_name, split = val_dataset.split("|")
        images, labels, qas = _find_liss4reason_triplets(
            base_image_dir, ds_name, split, images_dirname, labels_dirname, qas_dirname
        )
        self.images = images
        self.labels = labels
        self.qas = qas
        self.data_type = "liss4_reason"

        print(f"number of LISS4Reason {split} samples: ", len(images))

    def __len__(self):
        return len(self.images)

    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label_path = self.labels[idx]
        qa_path = self.qas[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise FileNotFoundError(f"Could not read mask: {label_path}")
        binary_mask = (raw_mask != 0).astype(np.uint8)  # (H, W), 0/1

        with open(qa_path, "r") as f:
            qa = json.load(f)
        questions_pool = qa.get("questions", [])
        question_text = questions_pool[0].strip() if questions_pool else \
            "What is the target region in this image?"
        sampled_sents = [question_text]

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        _seg = "[SEG]" if self.seg_token_num == 1 else \
            ' '.join(["[SEG{}]".format(i) for i in range(self.seg_token_num)])

        conv.messages = []
        conv.append_message(
            conv.roles[0],
            DEFAULT_IMAGE_TOKEN + "\n {} Please output segmentation mask.".format(question_text),
        )
        conv.append_message(conv.roles[1], "{}.".format(_seg))
        conversations.append(conv.get_prompt())

        if self.pad_val_clip_images:
            image_clip = self.transform_clip.apply_image(image)
            clip_resize = image_clip.shape[:2]
            image_clip = self.preprocess(
                torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(),
                self.clip_image_processor.size['shortest_edge'],
            )
        else:
            image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")[
                "pixel_values"
            ][0]
            clip_resize = image_clip.shape[-2:]

        image = self.transform.apply_image(image)
        resize = image.shape[:2]
        image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous(), self.img_size)

        masks = np.stack([binary_mask], axis=0)
        masks = torch.from_numpy(masks)
        labels = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label
        inference = True

        if self.masks_process_with_clip:
            mask_shape = image_clip.shape[-1]
            masks = transform_mask(masks, mask_shape)

        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            clip_resize,
            None,
            None,
            False,
            inference,
        )