from datetime import datetime

import polars as pl
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.classification_model.metrics import get_metric
from src.classification_model.utils import save_checkpoint, EpochsDataStorage

# TODO update with torchmetrics


def training(
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    loss_function: torch.nn.modules.loss,
    optimizer: torch.optim.Optimizer,
    start_epoch: int,
    config: dict,
    device: torch.device,
):
    """
    Train the model on the provided data.

    Args:
        train_loader (torch.utils.data.DataLoader): The training dataset loader.
        test_loader (torch.utils.data.DataLoader): The testing dataset loader.
        model (torch.nn.Module): The neural network model to be trained.
        loss_function (torch.nn.Module): The loss function used for training.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        start_epoch (int): The starting epoch number.
        config (dict): A dictionary containing the configuration for the run.
        device (str or torch.device): The device to be used for training.

    Returns:
        cpkt (dict): A dictionary containing the checkpoint information.
        metrics (pl.DataFrame): A Polars DataFrame containing the metrics.
    """

    # Create a Tensorboard SummaryWriter
    writer = SummaryWriter(f"runs/{config['name']}_{config['id_config']}")

    # Initialize checkpoint dictionary and metrics list
    cpkt = {}
    epochs_data = EpochsDataStorage()

    # Initialize best test value
    best_test_value = None

    for epoch in range(start_epoch, config["epochs"]):
        # set models to train mode
        model.train()

        epochs_data.reset_epoch_data()

        # Initialize loss
        train_loss = 0
        test_loss = 0

        # Iterate over the training dataset
        for _, data in tqdm(enumerate(train_loader), total=len(train_loader)):
            # Data preparation
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            label = data["targets"].to(device, dtype=torch.float)

            # Forward pass
            out = model(ids, mask)

            # Backward pass
            loss = loss_function(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate training loss and store predictions for metrics calculation
            train_loss += loss
            epochs_data.add_train_data(label, out)

        # Calculate metrics for the epoch
        train_true, train_pred = epochs_data.get_train_data()
        train_metrics_df, train_metrics_str = _calculate_metrics(
            train_true, train_pred, config["metrics"], "train", epoch
        )

        # Append to metrics list and write to Tensorboard
        epochs_data.add_metrics(train_metrics_df)
        writer.add_scalar("Loss/Train", train_loss / len(train_loader), epoch)
        print(
            f"Epoch:{epoch} --Train-- loss:{train_loss / len(train_loader):.3f} {train_metrics_str}"
        )

        # Evaluate model on test set every 'test_interval' epochs
        if epoch % config.get("test_interval", 1) == 0:
            model.eval()

            with torch.no_grad():
                # Iterate over the testing dataset
                for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                    # data preparation
                    ids = data["ids"].to(device, dtype=torch.long)
                    mask = data["mask"].to(device, dtype=torch.long)
                    label = data["targets"].to(device, dtype=torch.float)

                    out = model(ids, mask)

                    # Save metrics of the iteration
                    test_loss += loss_function(out, label)
                    epochs_data.add_test_data(label, out)

            # Calculate metrics for the test set
            test_true, test_pred = epochs_data.get_test_data()
            test_metrics_df, test_metrics_str = _calculate_metrics(
                test_true, test_pred, config["metrics"], "test", epoch
            )
            epochs_data.add_metrics(test_metrics_df)
            writer.add_scalar("Loss/Test", test_loss / len(test_loader), epoch)
            print(
                f"Epoch:{epoch} --Test--  loss={test_loss / len(test_loader):.3f} {test_metrics_str}"
            )

            # Save checkpoint if necessary
            best_test_value = _save_checkpoint_if_better(
                model, optimizer, cpkt, best_test_value, test_metrics_df, epoch, config
            )

    # Transform metrics to dataframe
    metrics = pl.concat(epochs_data.get_metrics())

    writer.close()
    return cpkt, metrics


def _calculate_metrics(
    labels: np.ndarray,
    outputs: np.ndarray,
    metrics: list,
    dataset_type: str,
    epoch: int,
):
    """
    Calculate the provided metrics for the given labels and outputs.

    Args:
        labels (np.ndarray): True labels.
        outputs (np.ndarray): Model predictions.
        metrics (list): List of metric names to calculate.
        dataset_type (str): Type of dataset ('train' or 'test').
        epoch (int): Current epoch number.

    Returns:
        tuple: A Polars DataFrame and a string representation of the calculated metrics.
    """
    metrics_list = []

    for metric_name in metrics:
        metric_value = get_metric(metric_name, labels, outputs, average="micro")
        metrics_list.append(
            {
                "epoch": epoch,
                "metric": metric_name,
                "value": metric_value,
                "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                "dataset": dataset_type,
            }
        )

    metrics_df = pl.DataFrame(metrics_list)
    metrics_string = " ".join(
        [
            f"-- {name}: {val:.3f}"
            for name, val in zip(metrics, metrics_df["value"].to_numpy())
        ]
    )

    return metrics_df, metrics_string


def _save_checkpoint_if_better(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cpkt: dict,
    best_value: float,
    test_metrics: pl.DataFrame,
    epoch: int,
    config: dict,
) -> float:
    """
    Save the checkpoint if the current value is better than the previous one.

    Args:
        model (torch.nn.Module): The neural network model.
        optimizer (torch.optim.Optimizer): The optimizer used for training.
        cpkt (dict): A dictionary containing the checkpoint information.
        best_value (float): The best value obtained so far.
        test_metrics (pl.DataFrame): A dictionary containing the metrics of the test set.
        config (dict): A dictionary containing the configuration for the run.

    """

    chosen_metric = config["metric_best_model"]
    metric_value = (
        test_metrics.filter(pl.col("metric") == chosen_metric)
        .get_column("value")
        .item()
    )

    # The comparaison with the best value is with >, therefor for the loss we need to do a modification
    if chosen_metric == "loss":
        metric_value *= -1

    if (best_value is None) or (metric_value > best_value):
        print("Saving model")
        cpkt = {
            "net": model.state_dict(),
            "epoch": epoch,
            "optim": optimizer.state_dict(),
        }
        save_checkpoint(
            cpkt, f"checkpoints/{config['name']}_{config['id_config']}.ckpt"
        )

        return metric_value

    return best_value
