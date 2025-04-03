import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_metric(
    metric: str,
    true_label: torch.Tensor,
    pred_label: torch.Tensor,
    pred_probas: torch.Tensor = torch.tensor(0.0),
    average: str = "binary",
) -> float:
    """
    Compute the specified metric for the given true labels and predictions.

    Args:
        metric (str): The name of the metric to compute.
            Supported metrics are:
                - "f1"
                - "recall"
                - "balanced_accuracy"
                - "accuracy"
                - "precision"
                - "auc" ( alias for average_precision_score)
                - "roc_auc"

        true (torch.Tensor): Ground truth labels.

        pred (torch.Tensor): Predicted labels.

        pred_probas (torch.Tensor, optional): Predicted probabilities.
            Defaults to torch.tensor(0.0).

        average (str, optional): Average method to use for multi-class metrics.
            Defaults to 'binary'.

    Returns:
        float: The computed metric value.

    Raises:
        NotImplementedError: If the specified metric is not supported.
    """

    # Define a dictionary mapping metric names to their corresponding functions
    metric_functions = {
        "f1": lambda true_label, pred_label: f1_score(true_label, pred_label, average=average),
        "recall": lambda true_label, pred_label: recall_score(true_label, pred_label, average=average),
        "balanced_accuracy": balanced_accuracy_score,
        "accuracy": accuracy_score,
        "precision": lambda true_label, pred_label: precision_score(true_label, pred_label, average=average),
        # Special handling for auc and roc_auc
        "auc": average_precision_score,
        "roc_auc": roc_auc_score,
    }

    # Check if the metric is supported
    if metric not in metric_functions:
        raise NotImplementedError(f"Metric {metric} is not yet supported...")

    # Handle auc and roc_auc separately since they require probabilities
    if metric in ["auc", "roc_auc"]:
        # Ensure pred_probas has shape (n_samples, 2)
        assert pred_probas.shape[1] == 2, "Pred probas must have shape (n_samples, 2)"
        proba = pred_probas[:, 1]
        return metric_functions[metric](true_label, proba)

    # Compute the metric using the corresponding function
    return metric_functions[metric](true_label, pred_label)
