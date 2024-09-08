import argparse
import os
import time
import numpy as np

from skimage.transform import resize


def crop_and_resize_reshape_image(img, is_overlay=False):
    reshaped_img = [None] * min(img.shape[0], 12)

    # Reduce to 12 slices.
    if img.shape[0] > 12:
        slice_index = abs(img.shape[0] - 12)
        img = img[:-slice_index]  # standardizing to 12 slices

    # Calculate crop size as a percentage of the image dimensions
    crop_percentage = 0.4  # 20% of the image dimensions
    crop_size_x = int(img.shape[1] * crop_percentage)
    crop_size_y = int(img.shape[2] * crop_percentage)
    print(f"crop_size_x: {crop_size_x}, crop_size_y: {crop_size_y}")

    center_x, center_y = img.shape[1] // 2, img.shape[2] // 2
    print(f"center_x: {center_x}, center_y: {center_y}")
    start_x, end_x = center_x - crop_size_x // 2, center_x + crop_size_x // 2
    start_y, end_y = center_y - crop_size_y // 2, center_y + crop_size_y // 2
    print(f"start_x: {start_x}, end_x: {end_x}, start_y: {start_y}, end_y: {end_y}")

    # Crop and resize to 84 x 84.
    for slc in range(img.shape[0]):
        cropped_slice = img[slc, start_x:end_x, start_y:end_y]

        if np.isnan(cropped_slice).any():
            raise ValueError(f"Cropped slice at index {slc} contains NaN values")

        if is_overlay:
            slc_res = resize(cropped_slice, (84, 84, 3))
        else:
            slc_res = resize(cropped_slice, (84, 84))

        reshaped_img[slc] = slc_res

    return np.array(reshaped_img)

def resize_reshape_image(img, is_overlay=False):
    reshaped_img = [None] * min(img.shape[0], 12)

    # Reduce to 12 slices.
    if img.shape[0] > 12:
        slice_index = abs(img.shape[0] - 12)
        img = img[:-slice_index]  # standardizing to 25 slices

    # Resize to 84 x 84.
    for slc in range(img.shape[0]):
        if is_overlay:
            slc_res = resize(img[slc], (84, 84, 3))
        else:
            slc_res = resize(img[slc], (84, 84))
        reshaped_img[slc] = slc_res

    return reshaped_img

def apply_prostate_mask(image_data, prostate_mask_path):
    prostate_mask = np.load(prostate_mask_path)
    if prostate_mask.dtype == bool:
        prostate_mask = prostate_mask.astype(np.uint8)
    prostate_mask = resize(prostate_mask, (84, 84))
    processed_volume = np.zeros_like(image_data)
    for j in range(image_data.shape[1]):
        altered_image = np.transpose(image_data, (2, 3, 0, 1))
        altered_image = altered_image[:, :, :, j]
        altered_image = np.mean(altered_image, axis=2)
        processed_volume[:, j, :, :] = altered_image * prostate_mask

    # Reconstruct the original form
    reconstructed_volume = np.transpose(processed_volume, (0, 2, 3, 1))
    return reconstructed_volume

# Start the timer.
start_time = time.time()

print("Starting script...")

# Create the parser
parser = argparse.ArgumentParser()

# Modify the arguments
parser.add_argument(
    "--raw_data_loc",
    help="Location of patient data folder",
    type=str,
    default="data",
)
parser.add_argument(
    "--modality1", 
    help="First modality to process", 
    type=str, 
    required=True
)
parser.add_argument(
    "--modality2", 
    help="Second modality to process", 
    type=str, 
    required=True
)
parser.add_argument(
    "--modality3", 
    help="Third modality to process", 
    type=str, 
    required=True
)
parser.add_argument(
    "--output_loc",
    help="Location to output files",
    type=str,
    default="data/multi_modal_images_nifti",
)

# Parse the arguments
args = parser.parse_args()

raw_data_loc = args.raw_data_loc
modalities = [args.modality1, args.modality2, args.modality3]
output_loc = args.output_loc

print("Starting script...")

# Make the output folder if it doesn't exist.
if not os.path.isdir(output_loc):
    os.makedirs(output_loc)

patient_ids = [f.name for f in os.scandir(raw_data_loc) if f.is_dir()]

# Counter to track number of null clinical_sig.
num_null_clinical_sig = 0

# Function to process a single modality
def process_modality(patient_loc, modality):
    if modality.startswith("DWIb"):
        b_index = int(modality[-1]) - 1
        modality_img = np.load(f"{patient_loc}/images/DWI.npy")[b_index, :, :, :]
        modality_img = np.transpose(modality_img, (2, 0, 1))
    else:
        modality_img = np.load(f"{patient_loc}/images/{modality}.npy")
        modality_img = np.transpose(modality_img, (2, 0, 1))
    
    modality_img = modality_img[..., None]
    modality_img = np.float64(modality_img)
    std_img = np.array(crop_and_resize_reshape_image(modality_img))
    std_img = std_img[np.newaxis, ...]
    return std_img

# Loop for all patients in file.
for patient_id in patient_ids:
    patient_loc = os.path.join(raw_data_loc, patient_id)
    print(f"Processing patient_id: {patient_id}")

    try:
        # Process each modality
        processed_images = []
        for modality in modalities:
            processed_img = process_modality(patient_loc, modality)
            processed_images.append(processed_img)

        # Concatenate the processed images
        final_img = np.concatenate(processed_images, axis=0)
        final_img = final_img[:, :, :, :, 0]
        print("final_img shape: ", final_img.shape)

        # Save npy array.
        output_filename = f"{patient_id}_multi_modal_img.npy"
        np.save(os.path.join(output_loc, output_filename), final_img)
        print(f"patient_id: {patient_id} saved")

    except Exception as e:
        print(f"Error processing patient_id: {patient_id}")
        print(f"Error message: {str(e)}")
        continue

# Stop the timer and print out result.
print("Script complete!")
print("--- %s seconds ---" % (time.time() - start_time))
