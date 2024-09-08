import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

MODALITIES = ["DWI", "T2W", "ADC"]

pairings = [[16, 54]]


def setup_image(image: np.array, slice: int, modality: int):
    data = np.transpose(image, (2, 3, 0, 1))
    data = data[modality, :, :, slice]
    normalized_image = (data - np.min(data)) / (np.max(data) - np.min(data)) * 255
    normalized_image = normalized_image.astype(np.uint8)
    normalized_image = cv2.resize(normalized_image, (84, 84))
    return normalized_image


def crop_lesion_mask(img: np.array, slice: int):

    crop_percentage = 0.4  # 20% of the image dimensions
    crop_size_x = int(img.shape[0] * crop_percentage)
    crop_size_y = int(img.shape[1] * crop_percentage)
    print(f"crop_size_x: {crop_size_x}, crop_size_y: {crop_size_y}")

    center_x, center_y = img.shape[0] // 2, img.shape[1] // 2
    print(f"center_x: {center_x}, center_y: {center_y}")
    start_x, end_x = center_x - crop_size_x // 2, center_x + crop_size_x // 2
    start_y, end_y = center_y - crop_size_y // 2, center_y + crop_size_y // 2
    print(f"start_x: {start_x}, end_x: {end_x}, start_y: {start_y}, end_y: {end_y}")

    cropped_slice = img[start_x:end_x, start_y:end_y, slice]
    cropped_slice = cv2.resize(cropped_slice, (84, 84))
    return cropped_slice


for i in pairings:
    patient_id = i[1]
    fold = i[0]
    SAVE_DIR = f"pca-results/attention_maps/3_modalities_v2/{patient_id}/individual_modalities"
    IMAGE_PATH = f"data/pca_processed_data/3_modalities_combined/ProstateX-{str(patient_id).zfill(4)}_multi_modal_img.npy"
    REAL_MASK_BASE_PATH = f"data/raw/ProstateX-{str(patient_id).zfill(4)}"
    lesion_mask = np.load(REAL_MASK_BASE_PATH + "/lesion_masks/T2w.npy")

    image_data = np.load(IMAGE_PATH)
    print(image_data.shape)  # (3, 12, 84, 84)

    # Set up the figure size and layout
    for modality in range(3):
        fig, axes = plt.subplots(3, 4, figsize=(15, 10))
        axes = axes.flatten()

        for slice in range(12):
            ax = axes[slice]
            ax.imshow(
                image_data[modality, slice, :, :], cmap="gray"
            )  # Overlay with mask
            ax.imshow(crop_lesion_mask(lesion_mask, slice), cmap="gray", alpha=0.5)
            ax.axis("off")
            ax.set_title(f"Slice {slice+1}")

        plt.tight_layout()
        os.makedirs(SAVE_DIR, exist_ok=True)
        plt.savefig(f"{SAVE_DIR}/image-with-lesion-mask-{MODALITIES[modality]}.png")
        plt.close(fig)
