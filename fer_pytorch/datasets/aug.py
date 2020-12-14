from albumentations import (
    Resize,
    RandomCrop,
    HorizontalFlip,
    Blur,GaussianBlur,MedianBlur,
    HueSaturationValue,
    RandomBrightnessContrast,
    IAASharpen,
    Normalize,
    OneOf, Compose,
    NoOp,
    GridDistortion
)

def  fer_train_aug(input_size, crop_residual_pix = 16):
    aug = Compose(
        [
            Resize(height = input_size + crop_residual_pix,
                   width  = input_size + crop_residual_pix),
            OneOf(
                [RandomCrop(height = input_size, width = input_size) ,
                 Resize(height=input_size,
                        width=input_size)
                 ],
            p = 1.0),
            HorizontalFlip(p = 0.5),
            GridDistortion(num_steps=5,distort_limit=0.1,p=0.5),
            # OneOf(
            #     [
            #         Blur(blur_limit = 7, p = 0.5),
            #         GaussianBlur(blur_limit = 7, p = 0.5),
            #         MedianBlur(blur_limit = 7, p = 0.5)
            #     ]
            # ),
            OneOf(
                [
                    HueSaturationValue(hue_shift_limit = 30,
                                       sat_shift_limit = 30,
                                       val_shift_limit = 30,
                                       p = 0.5),
                    RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3)
                ]
            ),
            #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            Normalize(mean=(0.485, 0.456, 0.406), std=(1.0/255, 1.0/255, 1.0/255))
        ],
    p = 1.0)

    return  aug


def  fer_test_aug(input_size):
    aug = Compose(
        [
            Resize(height = input_size,
                   width  = input_size),
            #Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            Normalize(mean=(0.485, 0.456, 0.406), std=(1.0 / 255, 1.0 / 255, 1.0 / 255))
        ],
    p = 1.0)
    return  aug