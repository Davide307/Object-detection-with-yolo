import json
import os
import shutil

import torch
from super_gradients import init_trainer, setup_device
from super_gradients.training import Trainer
from super_gradients.training import models
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, \
    coco_detection_yolo_format_val
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050, Accuracy, Top5
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback

##Converter da COCO format a YOLO
def converter():
    input_path = "datasets/FLIR_ADAS_v2/images_thermal_train/data"
    output_path = "datasets/FLIR_ADAS_v2/images_thermal_train/base_train/"
    f = open("datasets/FLIR_ADAS_v2/images_thermal_train/coco.json")
    data = json.load(f)
    f.close()
    file_names = []

    def load_images_from_folder(folder):
        count = 0
        for filename in os.listdir(folder):
            source = os.path.join(folder, filename)
            destination = f"{output_path}images/img{count}.jpg"

            try:
                shutil.copy(source, destination)
            # If source and destination are same
            except shutil.SameFileError:
                print("Source and destination represents the same file.")

            file_names.append(filename)
            count += 1

    load_images_from_folder(input_path)

    def get_img_ann(image_id):
        img_ann = []
        isFound = False
        for ann in data['annotations']:
            if ann['image_id'] == image_id:
                img_ann.append(ann)
                isFound = True
        if isFound:
            return img_ann
        else:
            return None

    def get_img(filename):
        for img in data['images']:
            # print(f"data/{filename}")
            if img['file_name'] == f"data/{filename}":
                return img

    count = 0

    for filename in file_names:
        # Extracting image
        img = get_img(filename)
        img_id = img['id']
        img_w = img['width']
        img_h = img['height']

        # Get Annotations for this image
        img_ann = get_img_ann(img_id)

        if img_ann:
            # Opening file for current image
            file_object = open(f"{output_path}labels/img{count}.txt", "a")

            for ann in img_ann:
                current_category = ann['category_id'] - 1  # As yolo format labels start from 0
                current_bbox = ann['bbox']
                x = current_bbox[0]
                y = current_bbox[1]
                w = current_bbox[2]
                h = current_bbox[3]

                # Finding midpoints
                x_centre = (x + (x + w)) / 2
                y_centre = (y + (y + h)) / 2

                # Normalization
                x_centre = x_centre / img_w
                y_centre = y_centre / img_h
                w = w / img_w
                h = h / img_h

                # Limiting upto fix number of decimal places
                x_centre = format(x_centre, '.6f')
                y_centre = format(y_centre, '.6f')
                w = format(w, '.6f')
                h = format(h, '.6f')

                # Writing current object
                file_object.write(f"{current_category} {x_centre} {y_centre} {w} {h}\n")

            file_object.close()
        count += 1  # This should be outside the if img_ann block.

#Copia le classi in un file formato YOLO + restituisce una lista con i nomi, necessaria per YOLO NAS
def getClasses(data):
    names = []
    for label in data['categories']:
        names.append(label.get("name"))
    with open('classes.txt', 'w') as file:
        i=0
        for item in names:
            # write each item on a new line
            file.write(f"{item}\n")
            i+=1
    return names

def test_model(model,test_root,destination):
    for img in os.listdir(test_root):
        res=model.predict(f"{test_root}/{img}")
        f = open(f"{destination}/{os.path.splitext(img)[0]}.txt", "w")
        for image_prediction in res:
            labels = image_prediction.prediction.labels
            confidence = image_prediction.prediction.confidence
            bboxes = image_prediction.prediction.bboxes_xyxy

            for i, (label, conf, bbox) in enumerate(zip(labels, confidence, bboxes)):
                f.write(label.astype(int).astype(str))
                f.write(" ")
                f.write(conf.astype(str))
                f.write(" ")
                rounded = bbox.astype(int)
                f.write(rounded[0].astype(str))
                f.write(" ")
                f.write(rounded[1].astype(str))
                f.write(" ")
                f.write(rounded[2].astype(str))
                f.write(" ")
                f.write(rounded[3].astype(str))
                f.write("\n")
        f.close()
#Setup per fare training (vari dataloaders e parametri per il train)
def trainsetup(dataset_params,model):
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['train_images_dir'],
            'labels_dir': dataset_params['train_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': 8,
            'num_workers': 2
        }
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['val_images_dir'],
            'labels_dir': dataset_params['val_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': 8,
            'num_workers': 2
        }
    )

    train_params = {
        # ENABLING SILENT MODE
        'silent_mode': False,
        "average_best_models": True,
        "warmup_mode": "linear_epoch_step",
        "warmup_initial_lr": 1e-6,
        "lr_warmup_epochs": 3,
        "initial_lr": 5e-4,
        "lr_mode": "cosine",
        "cosine_final_lr_ratio": 0.1,
        "optimizer": "Adam",
        "optimizer_params": {"weight_decay": 0.0001},
        "zero_weight_decay_on_bias_and_bn": True,
        "ema": True,
        "ema_params": {"decay": 0.9, "decay_type": "threshold"},
        "max_epochs": 20,
        "mixed_precision": True,
        "loss": PPYoloELoss(
            use_static_assigner=False,
            # NOTE: num_classes needs to be defined here
            num_classes=len(dataset_params['classes']),
            reg_max=16
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            )
        ],
        "metric_to_watch": 'mAP@0.50'
    }

    trainer.train(model=model,
                  training_params=train_params,
                  train_loader=train_data,
                  valid_loader=val_data)
def evaluateNAS(model,dataset_params):
    test_data = coco_detection_yolo_format_val(
        dataset_params={
            'data_dir': dataset_params['data_dir'],
            'images_dir': dataset_params['test_images_dir'],
            'labels_dir': dataset_params['test_labels_dir'],
            'classes': dataset_params['classes']
        },
        dataloader_params={
            'batch_size': 8,
            'num_workers': 2
        }
    )
    test_results=trainer.test(model=model,test_loader=test_data,test_metrics_list=DetectionMetrics_050(
                score_thres=0.1,
                top_k_predictions=300,
                include_classwise_ap=True,
                # NOTE: num_classes needs to be defined here
                num_cls=len(dataset_params['classes']),
                class_names=dataset_params['classes'],
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=0.01,
                    nms_top_k=1000,
                    max_predictions=300,
                    nms_threshold=0.7
                )
            ))
    print(test_results)
    

if __name__ == '__main__':
    #converter()
    #print("convertion completed")
    #le classi vengono caricate da getClasses (che prende da formato COCO e le mette in formato Yolo)
    data=json.load(open('datasets/FLIR_ADAS_v2/video_thermal_test/coco.json'))
    classes=getClasses(data)
    init_trainer()
    print(torch.cuda.is_available())
    trainer = Trainer(experiment_name='fine_tuning', ckpt_root_dir='save/fine_tuning')
    dataset_params = {
        'data_dir': 'datasets/FLIR_ADAS_v2',
        'train_images_dir': 'images_thermal_train/converted_train/images',
        'train_labels_dir': 'images_thermal_train/converted_train/labels',
        'val_images_dir': 'images_thermal_val/converted_val/images',
        'val_labels_dir': 'images_thermal_val/converted_val/labels',
        'test_images_dir': 'video_thermal_test/converted_test/images',
        'test_labels_dir': 'video_thermal_test/converted_test/labels',
        'classes': classes
    }
    model = models.get('yolo_nas_l',
                       num_classes=len(dataset_params['classes']),
                       checkpoint_path="save/fine_tuning/fine_tuning/RUN_base6to8new20/ckpt_best.pth")
    #trainsetup(dataset_params,model)
    evaluateNAS(model,dataset_params)
    #test_model(model,'FLIR_ADAS_v2/video_thermal_test/converted_test/images','save/fine_tuning/predictions 20122023')