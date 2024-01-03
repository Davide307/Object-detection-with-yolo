# Object-detection-with-yolo
## Questa repository contiene il codice utilizzato per fare train sul dataset FLIR_ADAS con yolo NAS e yolo 8, contiene anche il codice necessario per trasformare un dataset da formato COCO a YOLO su cui opera il seguente progetto
### main.py contiene il codice per fare training con YOLO-NAS
### yolo8.py contiene il codice per fare training con YOLO8
### augmentation.py consente di fare augmentation su questo particolare dataset con Albumentation
### box_augmentations_transformer.py contiene varie trasformazioni
### thermal_y8.yaml contiene i dati per il trainer yolo8
<sub>Le varie directory devono essere riempostate a seconda del progetto in locale, assicurasi che il dataset che sia sotto la directory datasets </sub>

