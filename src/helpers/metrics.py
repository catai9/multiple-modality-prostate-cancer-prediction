from sklearn.metrics import auc, roc_curve, RocCurveDisplay

import numpy as np


class Metrics:
    """Collects model predictions and targets to compute overall metrics"""

    def __init__(self):
        self.reset()

    def update(self, predictions, targets, loss=None):
        self.targets.append(targets.cpu().numpy().ravel())
        self.predictions.append(predictions.detach().cpu().numpy().ravel())
        if loss is not None:
            self.loss_sum += loss.item()
            self.loss_count += 1

    def reset(self):
        self.targets = []
        self.predictions = []
        self.loss_sum = 0
        self.loss_count = 0

    def evaluate(self, threshold=0.5, plot_roc=False):
        # Collect results
        targets = np.concatenate(self.targets)
        predictions = np.concatenate(self.predictions)

        # Compute AUC and plot ROC curve
        fpr, tpr, thresholds = roc_curve(targets, predictions, pos_label=1)
        auc_score = auc(fpr, tpr)

        # Make confusion matrix
        targets = targets.astype(np.uint8)
        predictions = np.uint8(predictions > threshold)
        stacked = np.stack((targets, predictions), axis=-1)
        unique, counts = np.unique(stacked, return_counts=True, axis=0)
        conf_matrix = np.zeros((2, 2))
        conf_matrix[unique[:, 0], unique[:, 1]] = counts

        # Compute CM-based metrics
        diag = np.diag(conf_matrix)
        accuracy = diag.sum() / max(conf_matrix.sum(), 1)
        specificity, sensitivity = diag / np.maximum(conf_matrix.sum(axis=1), 1)
        npv, ppv = diag / np.maximum(conf_matrix.sum(axis=0), 1)

        metrics = {
            "AUC": auc_score,
            "accuracy": accuracy,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "PPV": ppv,
            "NPV": npv,
        }

        if self.loss_count:
            metrics["loss"] = self.loss_sum / self.loss_count

        # Plot ROC curve
        if plot_roc:
            roc_display = RocCurveDisplay(
                fpr=fpr, tpr=tpr, roc_auc=auc_score, pos_label="Malignant"
            )
            roc_display.plot()
            return metrics, conf_matrix, roc_display

        return metrics, conf_matrix
