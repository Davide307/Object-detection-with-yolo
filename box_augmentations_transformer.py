import albumentations as A

def transforms_preset_1(min_visibility):
    preset = A.Compose([
        A.RandomRotate90(p=0.4),
        A.Flip(p=0.5),
        A.BBoxSafeRandomCrop( p=1.0),
        A.RandomBrightnessContrast(p=0.4),
    ], bbox_params=A.BboxParams(
        format='coco', label_fields=['class_labels'],
        min_visibility=min_visibility,min_area=1)
    )
    return preset


def transforms_preset_2(min_visibility):
    preset = A.Compose([
        A.RandomRotate90(p=0.4),
        A.Flip(p=0.5),
        A.BBoxSafeRandomCrop(p=6.0),
        A.RandomBrightnessContrast(p=0.4),
        A.CLAHE(p=0.6),
    ], bbox_params=A.BboxParams(
        format='coco', label_fields=['class_labels'],
        min_visibility=min_visibility,min_area=1)
    )
    return preset


def transforms_preset_3(min_visibility):
    preset = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.CLAHE(p=0.5),
        A.Flip(p=0.6),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.RandomFog(p=0.5),
    ], bbox_params=A.BboxParams(
        format='coco', label_fields=['class_labels'],
        min_visibility=min_visibility,min_area=1)
    )
    return preset


def transforms_preset_4(min_visibility):
    preset = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.Blur(p=0.5),
        A.RandomFog(p=0.5),
        A.CLAHE(p=0.5),
        A.RandomRotate90(p=0.5),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.RandomGamma(p=0.5)
    ], bbox_params=A.BboxParams(
        format='coco', label_fields=['class_labels'],
        min_visibility=min_visibility)
    )
    return preset


def transforms_preset_5(min_visibility):
    preset = A.Compose([
        A.BBoxSafeRandomCrop(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.5),
        A.RandomGamma(p=0.5)
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'],
                                min_visibility=min_visibility)
    )
    return preset


def transforms_preset_6(min_visibility):
    preset = A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.BBoxSafeRandomCrop(p=2.0),
        A.VerticalFlip(p=0.5)

    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'],
                                min_visibility=min_visibility)
    )
    return preset


def transforms_preset_7(min_visibility):
    preset = A.Compose([
        A.CLAHE(p=0.5),
        A.RandomRotate90(p=0.3),
        A.Flip(p=0.3),
        A.BBoxSafeRandomCrop(p=2.0),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.4),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'],
                                min_visibility=min_visibility)
    )
    return preset


def transforms_preset_8(min_visibility):
    preset = A.Compose([
        A.RGBShift(p=0.6),
        A.Blur(p=0.4),
        A.GaussNoise(p=0.4),
        A.Flip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.RandomBrightnessContrast(p=0.4)

    ], bbox_params=A.BboxParams(format='coco', label_fields=['class_labels'],
                                min_visibility=min_visibility)
    )
    return preset


TRANSFORMS_LIST = [transforms_preset_6,transforms_preset_7,transforms_preset_8]
