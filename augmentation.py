import json
import os
from itertools import repeat
from multiprocessing import Pool, pool

import albumentations as A
import cv2
import numpy as np
from pycocotools.coco import COCO

import box_augmentations_transformer as trasf

root_labels = 'datasets/FLIR_ADAS_v2/images_thermal_train/base_train/labels'
root_images = 'datasets/FLIR_ADAS_v2/images_thermal_train/base_train/images'
def parse_yolo_annotation(line):
    class_id, x_center, y_center, width, height = map(float, line.split())
    if (x_center > 1 or y_center > 1 or width > 1 or height > 1):
        print("errore")
    return [
        x_center,
        y_center,
        width,
        height
    ]


def parse_yolo_classes(line):
    class_id, x_center, y_center, width, height = map(float, line.split())
    int(class_id)
    return class_id


def parse_file(file):
    # Using readlines()
    Lines = file.readlines()
    bboxes = []
    classes = []
    # Strips the newline character
    for line in Lines:
        bboxes.append(parse_yolo_annotation(line))
        classes.append(parse_yolo_classes(line))
    return bboxes, classes


def transofrmpool(label,transforms,transform_ids):
    file = open(f"{root_labels}/{label}", 'r')
    bboxes, class_labels = parse_file(file)
    file.close()
    image = cv2.imread(f"{root_images}/{os.path.splitext(label)[0]}.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    coco_bboxes = bbox_yolo_to_coco(bboxes)
    for transform, transform_id in zip(transforms, transform_ids):
        transformed = (transform(0.1))(image=image, bboxes=coco_bboxes, class_labels=class_labels)
        yolo_bboxes = bbox_coco_to_yolo(transformed["bboxes"], transformed["image"])
        os.chdir('datasets/FLIR_ADAS_v2/images_thermal_train/converted_train/images')
        cv2.imwrite(f"{os.path.splitext(label)[0]}_{transform_id}.jpg", transformed["image"])
        os.chdir('../../../../..')
        with open(
                f"datasets/FLIR_ADAS_v2/images_thermal_train/converted_train/labels/{os.path.splitext(label)[0]}_{transform_id}.txt", "w") as f:
            for i, (bbox, class_id) in enumerate(zip(yolo_bboxes, transformed["class_labels"])):
                f.write(f"{int(class_id)} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")
        f.close()


def bbox_yolo_to_coco(bboxes):
    coco_bboxes = []
    width = 640
    height = 512
    for bbox in bboxes:
        coco_bbox = []
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]
        x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
        x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
        coco_bbox.append(x_min)
        coco_bbox.append(y_min)
        coco_bbox.append(x_max - x_min)
        coco_bbox.append(y_max - y_min)
        coco_bboxes.append(coco_bbox)
    return coco_bboxes


def bbox_coco_to_yolo(bboxes,image):
    yolo_bboxes = []
    for bbox in bboxes:
        yolo_bbox = []
        x = bbox[0]
        y = bbox[1]
        w = bbox[2]
        h = bbox[3]

        # Finding midpoints
        x_centre = (x + (x + w)) / 2
        y_centre = (y + (y + h)) / 2
        H, W, c = image.shape
        img_w = W
        img_h = H
        # Normalization
        x_centre = x_centre / img_w
        y_centre = y_centre / img_h
        w = w / img_w
        h = h / img_h
        yolo_bbox.append(x_centre)
        yolo_bbox.append(y_centre)
        yolo_bbox.append(w)
        yolo_bbox.append(h)
        yolo_bboxes.append(yolo_bbox)
    return yolo_bboxes


def get_bboxes_yolo():
    #la lista i serve per dare nomi distinti alle varie immagini trasformate
    print(len(trasf.TRANSFORMS_LIST))
    i = range(1,len(trasf.TRANSFORMS_LIST)+1)
    labels = os.listdir(root_labels)
    #cambiare gli elementi in trasf.TRANSFORMS_LIST a seconda dei trasform preset che si voglionmo usare
    with Pool() as pool:
        pool.starmap(transofrmpool,zip(labels,repeat(trasf.TRANSFORMS_LIST),repeat(i)))
if __name__ == '__main__':
    get_bboxes_yolo()
