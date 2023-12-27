from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resize
)
from torch.utils.data import Dataset, DataLoader

import numpy as np

from PIL import Image


def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))

        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 0, 0] # if there is no mask in the array, set bbox to 0

class SAMDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor, IMG_SIZE, ROI=(224,224)):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor
        self.IMG_SIZE = IMG_SIZE
        self.ROI = ROI
        self.transforms = transforms = Compose([

            # load .nii or .nii.gz files
            LoadImaged(keys=['img', 'label']),

            # add channel id to match PyTorch configurations
            EnsureChannelFirstd(keys=['img', 'label']),
            # ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),

            Orientationd(keys=["img", "label"], axcodes="RAS"),
            SpatialPadd(keys=["img", "label"], spatial_size=IMG_SIZE),

            Spacingd(keys=["img", "label"], pixdim=(
                1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
            RandScaleIntensityd(keys="img", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="img", offsets=0.1, prob=1.0),

            # CropForegroundd(keys=["img", "label"], source_key="img"),

            # EnsureTyped(keys=["img", "label"]),

            # Spacingd(
            #     keys=["img", "label"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # RandSpatialCropd(keys=["img", "label"], roi_size=ROI, random_size=False),
            # RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=0),
            # RandFlipd(keys=["img", "label"], prob=0.5, spatial_axis=1),

            # Orientationd(keys=["img", "label"], axcodes="RAS"),

            # RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            # NormalizeIntensityd(keys="img", nonzero=True, channel_wise=True),
            # RandScaleIntensityd(keys="img", factors=0.1, prob=1.0),
            # RandShiftIntensityd(keys="img", offsets=0.1, prob=1.0),


            # # reorient images for consistency and visualization
            # Orientationd(keys=['img', 'label'], axcodes='RA'),

            # # resample all training images to a fixed spacing
            # Spacingd(keys=['img', 'label'], pixdim=(1.5, 1.5), mode=("bilinear", "nearest")),

            # # consider CHANGING roi size
            # CenterSpatialCropd(keys=['img', 'label'], roi_size=IMG_SIZE),


            # ScaleIntensityRanged(keys=['label'], a_min=0, a_max=255,
            #              b_min=0.0, b_max=1.0, clip=True),


#             RepeatChanneld(keys=['img'], repeats=3, allow_missing_keys=True)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # create a dict of images and labels to apply Monai's dictionary transforms
        data_dict = self.transforms({'img': image_path, 'label': mask_path})

        # squeeze extra dimensions
        image = data_dict['img'].squeeze()
        ground_truth_mask = data_dict['label'].squeeze()

        # print("dims of image and mask:", image.shape, ground_truth_mask.shape)

        # convert to int type for huggingface's models expected inputs
        image = image.astype(np.uint8)

        # convert the grayscale array to RGB (3 channels)
        array_rgb = np.dstack((image, image, image)) # is this necessary??

        # convert to PIL image to match the expected input of processor
        image_rgb = Image.fromarray(array_rgb)

        # get bounding box prompt (returns xmin, ymin, xmax, ymax)
        # I will only be trying to get the regular segmented mask with all tumor classes (anything that is not 0)
        # merge 2 and 4 to construct tumor core, merge labels 1, 2 and 4 to construct WT, label 2 is enhancing tumor
        # skips label 3 for some reason

        """
        Annotations comprise the GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2), and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
        """

        ground_truth_mask = ground_truth_mask.astype(np.uint8)

        # print(np.unique(ground_truth_mask))

        # ground_truth_mask = np.where(ground_truth_mask == 0, 1, 0)



        prompt = get_bounding_box(ground_truth_mask)

        # prepare image and prompt for the model
        inputs = self.processor(image_rgb, input_boxes=[[prompt]], return_tensors="pt")

        # remove batch dimension which the processor adds by default
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # add ground truth segmentation (ground truth image size is 256x256)

        inputs["ground_truth_mask"] = torch.from_numpy(ground_truth_mask.astype(np.int8))

        return inputs