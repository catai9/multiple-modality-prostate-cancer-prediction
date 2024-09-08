import torch
from medcam import medcam
from get_model import get_trained_model
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

BACKENDS = ["gcampp", "gcam", "gbp", "ggcam"]
CSV_FILE_PATH = (
    "val_results.csv"
)
backend = "gcampp"
THRESHOLD = 40


def process_prostate_data(file_path):
    true_positive = []
    false_positive = []
    false_negative = []
    true_negative = []

    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row

        for row in reader:
            index = int(row[0]) + 1  # Add 1 to match the fold number
            patient_id = row[1].split("-")[1]  # Extract the number part
            if patient_id == "0000":
                patient_id = 0
            else:
                patient_id = patient_id.lstrip("0")
            y_pred = int(row[2])
            y_true = int(row[3].strip("[]"))  # Remove brackets and convert to int

            if y_pred == 1 and y_true == 1:
                true_positive.append(["true_positive", index, patient_id])
            elif y_pred == 1 and y_true == 0:
                false_positive.append(["false_positive", index, patient_id])
            elif y_pred == 0 and y_true == 1:
                false_negative.append(["false_negative", index, patient_id])
            elif y_pred == 0 and y_true == 0:
                true_negative.append(["true_negative", index, patient_id])

    return [true_positive, false_positive, false_negative, true_negative]


def setup_image(image: np.array, slice: int):
    data = np.transpose(image, (2, 3, 0, 1))
    data = data[:, :, :, slice]
    normalized_image = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_image


pairings = process_prostate_data(CSV_FILE_PATH)

summed_attention = None

for pairing in pairings:
    for i in pairing:
        name = i[0]
        fold = i[1]
        patient_id = i[2]
        MODEL_PATH = f"pca-results/no_freeze_monai_resnet_34_23_T2w_only_prostate_clinical_sig/best_weights/best_metric_model_fold_{fold}.pth"
        IMAGE_PATH = f"data/pca_processed_data/T2w/ProstateX-{str(patient_id).zfill(4)}_T2w_img.npy"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_trained_model(MODEL_PATH)
        model = model.to(device)
        model = medcam.inject(
            model,
            output_dir="attention_maps",
            save_maps=True,
            backend=backend,
            layer="auto",
            replace=True,
        )
        model.eval()

        image_data = np.load(IMAGE_PATH)
        new_input = torch.from_numpy(np.array([image_data])).float().to(device)

        output = model(new_input)
        output_np = output.cpu().numpy()
        output_np = (output_np - np.min(output_np)) / (
            np.max(output_np) - np.min(output_np)
        )  # normalizing

        # Sum all slices for this patient
        patient_summed_output = np.sum(output_np[0, 0], axis=0)

        # Add to the total sum
        if summed_attention is None:
            summed_attention = patient_summed_output
        else:
            summed_attention += patient_summed_output

        print(f"Processed patient {patient_id}")

    # Normalize the final summed attention map
    normalized_summed_attention = (summed_attention - np.min(summed_attention)) / (
        np.max(summed_attention) - np.min(summed_attention)
    )

    # Create a heatmap of the summed attention
    plt.figure(figsize=(10, 10))
    plt.imshow(normalized_summed_attention, cmap="jet", interpolation="nearest")
    plt.colorbar(label="Normalized Summed Attention")
    plt.title(f"Summed Attention Map Across All Patients")
    plt.axis("off")

    # Save the heatmap
    SAVE_DIR = "pca-results/prostate_only_model/attention_maps"
    os.makedirs(SAVE_DIR, exist_ok=True)
    plt.savefig(f"{SAVE_DIR}/{backend}_{name}_heatmap.png")
    plt.close()

    # Create a binary mask of the highest values (e.g., top 10%)
    threshold = np.percentile(normalized_summed_attention, THRESHOLD)
    high_value_mask = normalized_summed_attention > threshold

    plt.figure(figsize=(10, 10))
    plt.imshow(high_value_mask, cmap="gray")
    plt.title(f"High Value Areas Across All Patients (Top {100 - THRESHOLD}%)")
    plt.axis("off")

    # Save the binary mask
    plt.savefig(f"{SAVE_DIR}/{backend}_{name}_mask_{THRESHOLD}.png")
    plt.close()

    print("Processing complete. Final summed mask created.")

    np.save(f"{SAVE_DIR}/{backend}_{name}_{THRESHOLD}.npy", high_value_mask)
