from easydict import EasyDict as edict
from datetime import datetime

import time
import argparse
import pandas as pd
import yaml
import os

from helpers.process_data import process_data
from helpers.train_model import setup_train_model


# Start the timer and print out log that script started.
start_time = time.time()

# Create the parser
parser = argparse.ArgumentParser()

# Read in arguments for input.
parser.add_argument(
    "--model-config", help="Location of the model config file", type=str, required=True
)
parser.add_argument(
    "--modality", help="Image modality for training", type=str, required=True
)
parser.add_argument("-gi", "--gpu-id", type=int, default=2, help="GPU device ID")
parser.add_argument("-bs", "--batch-size", type=int, default=4, help="Batch size to use")
parser.add_argument(
    "--use_pretrained_sbr_model", action="store_true", help="Whether to use the best pretrained SBR grade model"
)

# Parse the arguments
args = parser.parse_args()

with open(args.model_config, "r") as f:
    hparams = yaml.load(f, Loader=yaml.FullLoader)

config = edict(hparams)

modality = args.modality
use_pretrained_sbr_model = args.use_pretrained_sbr_model
batch_size = args.batch_size
image_dir = config.image_dir + f"/{modality}"
metadata_loc = config.metadata_loc
path_to_weights = config.path_to_weights
output_dir = config.output_dir + f"{modality}"
server_name = config.server_name
num_samples = config.num_samples
total_epochs = config.total_epochs
is_train_transform = config.is_train_transform
learning_rate = config.learning_rate

# Make the output folder if it doesn't exist.
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

# Set up log file to write logs for.
date_str = datetime.today().strftime("%Y-%m-%d")
log_file_name = (
    f"logs/{server_name}-{date_str}-{learning_rate}-fix-weights-train-{is_train_transform}-{use_pretrained_sbr_model}runs-{modality}.txt"
)

# Print out the values passed in for the arguments.
with open(log_file_name, "a") as f:
    f.write(f"Starting script to train model for {modality} images... \n")
    f.write(f"Argument values:\n")
    f.write(f"- image_dir = {image_dir}\n")
    f.write(f"- metadata_loc = {metadata_loc}\n")
    f.write(f"- path_to_weights = {path_to_weights}\n")
    f.write(f"- output_dir = {output_dir}\n")
    f.write(f"- server_name = {server_name}\n")
    f.write(f"- modality = {modality}\n")
    f.write(f"- num_samples = {num_samples}\n")

device = "cuda:{}".format(args.gpu_id) if args.gpu_id >= 0 else "cpu"

# Process data.
with open(log_file_name, "a") as f:
    f.write(f"Step 1: Processing data... \n")

df_metadata = pd.read_csv(metadata_loc)
patient_ids, patient_images, patient_labels = process_data(
    image_dir, df_metadata, log_file_name, modality
)

# Setup the model.
with open(log_file_name, "a") as f:
    f.write(f"Step 2: Setup model... \n")

# Train the model.
(
    val_patient_ids,
    y_pred,
    y_true,
    validation_accuracy,
    fpr,
    tnr,
    fnr,
    tpr,
) = setup_train_model(
    patient_ids,
    patient_images,
    patient_labels,
    num_samples,
    total_epochs,
    output_dir,
    log_file_name,
    is_train_transform,
    learning_rate,
    device,
    path_to_weights,
    batch_size=batch_size,
    use_pretrained_sbr_model=use_pretrained_sbr_model
)

# Save the model evaluation.
with open(log_file_name, "a") as f:
    f.write("Printing Results...\n")
    f.write(f"val_patient_ids: {val_patient_ids}\n")
    f.write(f"y_pred: {y_pred}\n")
    f.write(f"y_true: {y_true}\n")
    f.write(f"validation_accuracy: {validation_accuracy}\n")
    f.write(f"fpr: {fpr}\n")
    f.write(f"tnr: {tnr}\n")
    f.write(f"fnr: {fnr}\n")
    f.write(f"tpr: {tpr}\n")

df_results = pd.DataFrame(
    {
        "patient_ids": val_patient_ids,
        "y_pred": y_pred,
        "y_true": y_true,
        "validation_accuracy": validation_accuracy,
    }
)
df_results.to_csv(f"{output_dir}/val_results.csv")

# Stop the timer and print out time to run.
with open(log_file_name, "a") as f:
    f.write(f"Script to train model for {modality} images completed! \n")
    f.write("--- %s seconds ---" % (time.time() - start_time))
