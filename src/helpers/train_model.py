from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.model_selection import LeaveOneOut
from monai.data import ArrayDataset
from time import perf_counter

from helpers.get_model import get_model
from helpers.metrics import Metrics

import torch
import os


def get_fold_val_patient(val_patient_id, df_fold_mapping):
    return df_fold_mapping.loc[df_fold_mapping["patient_id"] == val_patient_id][
        "fold"
    ].values[0]


def setup_train_model(
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
    batch_size=4,
    use_pretrained_sbr_model=False,
):
    validation_accuracy = []
    val_patient_ids = []
    y_pred = []
    y_true = []
    current_fold = 1
    tpr = 0
    fpr = 0
    tnr = 0
    fnr = 0

    loo = LeaveOneOut()

    for train_index, val_index in loo.split(patient_images):
        val_patient_id = patient_ids[val_index[0]]
        print(f"Current Fold = {current_fold}")
        # Crop the image
        print(f"Val_index = {val_index}")
        print(f"Patient in valid is = {val_patient_id}")

        val_patient_ids.append(val_patient_id)
        x_train, x_valid = patient_images[train_index], patient_images[val_index]
        y_train, y_valid = patient_labels[train_index], patient_labels[val_index]

        if is_train_transform:
            transforms = torch.nn.Sequential(
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.5),
            )
            train_transforms = torch.jit.script(transforms)

            train_ds = ArrayDataset(
                img=torch.from_numpy(x_train),
                labels=y_train,
                img_transform=train_transforms,
            )
        else:
            train_ds = ArrayDataset(img=torch.from_numpy(x_train), labels=y_train)

        train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            weights=torch.tensor([200/130, 200/70], device=device),
            num_samples=num_samples,
            replacement=True,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=train_sampler, pin_memory=True
        )

        val_ds = ArrayDataset(img=torch.from_numpy(x_valid), labels=y_valid)
        val_loader = DataLoader(val_ds, batch_size=batch_size, pin_memory=True)

        # getting the pretrained model
        model = get_model(device, path_to_weights, use_pretrained_sbr_model)

        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs
        )

        best_result = train_model_fold(
            total_epochs,
            model,
            current_fold,
            train_loader,
            val_loader,
            loss_function,
            optimizer,
            scheduler,
            len(train_ds),
            output_dir,
            log_file_name,
            10,
            device,
        )

        validation_accuracy.append(best_result)
        y_true.append(y_valid)

        if y_valid == 0 and best_result == 0:
            # Negative and did not get it right.
            fpr += 1
            y_pred.append(1)
        elif y_valid == 0 and best_result == 1:
            # Negative and got it right.
            tnr += 1
            y_pred.append(0)
        elif y_valid == 1 and best_result == 0:
            # Positive and did not get it right.
            fnr += 1
            y_pred.append(0)
        elif y_valid == 1 and best_result == 1:
            # Positive and got it right.
            tpr += 1
            y_pred.append(1)
        else:
            print(
                f"Invalid result for y_valid = {y_valid}, best_results = {best_result}"
            )

        current_fold += 1

    return val_patient_ids, y_pred, y_true, validation_accuracy, fpr, tnr, fnr, tpr


def train_model_fold(
    epochs,
    model,
    fold,
    train_loader,
    val_loader,
    loss_function,
    optimizer,
    scheduler,
    train_ds_length,
    output_dir,
    log_file_name,
    log_interval,
    device,
):
    ckpt_dir = os.path.join(output_dir, "experiments/checkpoints")
    log_dir = os.path.join(output_dir, "experiments/events")
    best_weights_dir = os.path.join(output_dir, "best_weights")

    # Make the output folders if they don't exist.
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    if not os.path.isdir(best_weights_dir):
        os.makedirs(best_weights_dir)

    best_metric = -1
    best_metric_epoch = -1

    writer = SummaryWriter(log_dir)
    max_epochs = epochs

    train_metrics = Metrics()
    val_metrics = Metrics()

    for epoch in range(max_epochs):
        print("Epoch: ", epoch)
        epoch_time_start = perf_counter()

        # TRAIN: Train model on train set.
        model.train()

        y_pred_t = torch.tensor([], dtype=torch.float32, device=device)
        y_true_t = torch.tensor([], dtype=torch.long, device=device)
        train_step = 0

        for train_imgs, train_labels in train_loader:
            train_step += 1
            train_labels = train_labels.type(torch.LongTensor)  # casting to long
            train_imgs, train_labels = train_imgs.float().to(device), train_labels.to(
                device
            )
            optimizer.zero_grad()
            y_pred_train = model(train_imgs)
            y_pred_t = torch.cat([y_pred_t, y_pred_train.argmax(dim=1)], dim=0)
            y_true_t = torch.cat([y_true_t, train_labels], dim=0)
            loss = loss_function(y_pred_train, train_labels)
            loss.backward()
            optimizer.step()
            epoch_len = train_ds_length // train_loader.batch_size

            # Update train metrics
            writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + train_step)
            train_metrics.update(y_pred_t, y_true_t, loss=loss)

            # Log/print metrics
            if not train_step % log_interval:
                metrics, _ = train_metrics.evaluate()
                print(
                    "[epoch: {}, step: {}, loss: {}]".format(
                        epoch, train_step, metrics["loss"]
                    )
                )
                for key, val in metrics.items():
                    writer.add_scalar("train/" + key, val, global_step=train_step)
                train_metrics.reset()

        scheduler.step()

        train_acc_value = torch.eq(y_pred_t, y_true_t)
        train_acc_metric = train_acc_value.sum().item() / len(train_acc_value)

        writer.add_scalar("train_acc", train_acc_metric, epoch + 1)

        # VALID: Get results on validation set.
        model.eval()

        y_pred_v = torch.tensor([], dtype=torch.float32, device=device)
        y_true_v = torch.tensor([], dtype=torch.long, device=device)
        val_step = 0

        with torch.no_grad():
            for val_images, val_labels in val_loader:
                val_step += 1
                val_labels = val_labels.type(torch.LongTensor)  # casting to long
                val_images, val_labels = val_images.float().to(device), val_labels.to(
                    device
                )
                y_pred_val = model(val_images)
                y_pred_v = torch.cat([y_pred_v, y_pred_val.argmax(dim=1)], dim=0)
                y_true_v = torch.cat([y_true_v, val_labels], dim=0)
                loss = loss_function(y_pred_val, val_labels)
                writer.add_scalar(
                    "valid_loss", loss.item(), epoch_len * epoch + val_step
                )
                val_metrics.update(y_pred_v, y_true_v)

            valid_acc_value = torch.eq(y_pred_v, y_true_v)
            valid_acc_metric = valid_acc_value.sum().item() / len(valid_acc_value)

            # SAVE: Save best model (best on validation results).
            if valid_acc_metric > best_metric:
                best_metric = valid_acc_metric
                best_metric_epoch = epoch + 1

                torch.save(
                    model.state_dict(),
                    str(best_weights_dir)
                    + "/best_metric_model_fold_"
                    + str(fold)
                    + ".pth",
                )
            print(
                "Current epoch: {} current accuracy: {:.4f} best accuracy: {:.4f} at epoch {}".format(
                    epoch + 1, valid_acc_metric, best_metric, best_metric_epoch
                )
            )

        # Log validation metrics
        writer.add_scalar("val_acc", valid_acc_metric, epoch + 1)
        metrics, conf_matrix, roc_display = val_metrics.evaluate(plot_roc=True)
        for key, val in metrics.items():
            writer.add_scalar("val/" + key, val, global_step=train_step)
        fig = roc_display.figure_
        fig.set_size_inches(6, 6)
        writer.add_figure("val/roc_curve", fig, global_step=train_step)
        val_metrics.reset()
        writer.flush()

        # Save checkpoint
        ckpt_file = os.path.join(ckpt_dir, "checkpoint-{:04d}.pth".format(epoch))
        torch.save(model.state_dict(), ckpt_file)

        epoch_time_end = perf_counter()
        epoch_time = epoch_time_end - epoch_time_start

        with open(log_file_name, "a") as f:
            f.write(f"Epoch {epoch} time: {epoch_time}\n")

        if best_metric > 0:
            with open(log_file_name, "a") as f:
                f.write(f"Early stopping as the best metric is greater than 0!")
                f.write(
                    f"Training model done for fold {fold}, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch} \n"
                )

            writer.close()
            return best_metric

    with open(log_file_name, "a") as f:
        f.write(
            f"Training model done for fold {fold}, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch} \n"
        )

    writer.close()

    return best_metric
