import os

from ultralytics import YOLO
from ultralytics import settings
import numpy as np
import sys

#Testa il modello
def test_model(model, test_root, destination):
    results = model(f"{test_root}",device=1,conf=0.30)
    for result in results:
        boxes = result.boxes.cpu().numpy()
        with open(f"{destination}/{os.path.splitext(os.path.basename(result.path))[0]}.txt", 'w') as file:
            for box in boxes:
                file.write(f"{box.cls[0].astype(int)} {np.round(box.conf[0],2)} {box.xyxy[0][0].astype(int)} {box.xyxy[0][1].astype(int)} {box.xyxy[0][2].astype(int)} {box.xyxy[0][3].astype(int)}\n")




if __name__ == '__main__':
    model = YOLO('save/fine_tuning/fine_tuning/RUNbase6to8(new)')
    #test_model(model,'datasets/FLIR_ADAS_v2/video_thermal_test/converted_test/images','save_yolo8/aug6to8(new)base')
    #thermal_y8.yaml contiene le classi ed i path i vari dataset
    #model.train(data='thermal_y8.yaml', epochs=10, batch=10, device=1)
    met=model.val(data='thermal_y8_val.yaml',device=1)
    print(met.box.map50)