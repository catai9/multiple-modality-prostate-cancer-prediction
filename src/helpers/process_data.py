import numpy as np
import os
import math

def process_data(image_dir, df_metadata, model_type, log_file_name, modality):
    patient_ids = []
    patient_images = []
    patient_labels = []
    patient_file_list = os.listdir(image_dir)
    with open(log_file_name, "a") as f:
        f.write(
            f"Processing {model_type} data for {len(patient_file_list)} patients.\n"
        )
    for patient_file in patient_file_list:
        patient_id = patient_file.split("_")[0]
        patient_label = df_metadata.loc[df_metadata["patient_id"] == patient_id][
            model_type
        ]
        patient_label = patient_label.values[0]
        if math.isnan(patient_label) or patient_label == None or patient_label == "":
            print(f"skipping patient_id: {patient_id} as label Null")
            continue
        patient_image = np.load(f"{image_dir}/{patient_file}")
        patient_ids.append(patient_id)
        patient_images.append(patient_image)
        patient_labels.append(patient_label)

    assert len(patient_ids) == len(patient_labels)
    assert len(patient_ids) == len(patient_images)
    with open(log_file_name, "a") as f:
        f.write(f"Processed {model_type} data for {len(patient_ids)} patients.\n")
    patient_images = np.array(patient_images)
    patient_labels = np.array(patient_labels)
    return patient_ids, patient_images, patient_labels