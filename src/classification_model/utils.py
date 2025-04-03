import os
import shutil
import torch
import yaml
import numpy as np
import torch.nn as nn

from src.classification_model.optimizer import get_optimizer

from pathlib import Path
from src.classification_model.models import DefaultModel, Bert

def get_model(config: dict) -> nn.Module:
    """
    Returns a model instance based on the configuration.

    Args:
        config (dict): Model configuration dictionary.

    Returns:
        nn.Module: The instantiated model.
    """
    if config["model"] == "default":
        return DefaultModel()
    if config["model"] == "Bert":
        return Bert()
    else:
        raise NotImplementedError(f"Model {config['model']} is not yet supported")


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False) -> dict:
    """
    Loads a torch model from a checkpoint file.

    Args:
        ckpt_dir_or_file (str): Path to the checkpoint directory or filename.
        map_location: Can be used to directly load to a specific device.
        load_best (bool): If True, loads the 'best_model.ckpt' if exists.

    Returns:
        dict: The loaded checkpoint state dictionary.
    """
    # Check if ckpt_dir_or_file is a directory
    if os.path.isdir(ckpt_dir_or_file):
        # If loading best model, try to find it in the directory
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, 'best_model.ckpt')
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Best model checkpoint not found at {ckpt_path}")
        else:
            # Read latest_checkpoint.txt to find the latest checkpoint
            with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint.txt')) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        # If not a directory, assume it's a file path
        ckpt_path = ckpt_dir_or_file

    # Load the checkpoint using torch.load()
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(f'[*] Loading checkpoint from {ckpt_path} succeed!')
    return ckpt


def save_checkpoint(state: dict, save_path: str, is_best=False, max_keep=-1):
    """
    Saves a torch model to a checkpoint file.

    Args:
        state (dict): The state dictionary of the torch Neural Network.
        save_path (str): Destination path for saving the checkpoint.
        is_best (bool): If True, creates an additional copy as 'best_model.ckpt'.
        max_keep (int): Specifies the maximum number of checkpoints to keep.

    Returns:
        None
    """
    # Save the checkpoint using torch.save()
    torch.save(state, save_path)

    # Deal with max_keep by updating latest_checkpoint.txt
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, 'latest_checkpoint.txt')

    save_path = os.path.basename(save_path)
    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            ckpt_list = [save_path + '\n'] + ckpt_list
    else:
        ckpt_list = [save_path + '\n']

    # Remove old checkpoints based on max_keep
    if max_keep != -1:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    # Write updated latest_checkpoint.txt
    with open(list_path, 'w') as f:
        f.writelines(ckpt_list)

    # Copy the best model checkpoint
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, 'best_model.ckpt'))


def ensure_dir(dir_name: str):
    """
    Creates a directory if it does not exist.
    """
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def load_config_file(path):
    """
    Loads a YAML configuration file.
    """
    
    path = Path(path)
    with path.open(mode="r") as yamlfile:
        return yaml.safe_load(yamlfile)

def setup_device():
    """Set up the device for training (CPU or GPU)."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU")
    else:
        print("Using CPU")
    return device


def load_config_and_set_seed(config_path: dict):
    """Load the config file and set the seed for reproducibility."""
    config = load_config_file(config_path)
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    torch.backends.cudnn.benchmark = True
    return config


def load_model_and_optimizer(config: dict, device):
    """Load the model and optimizer."""
    model = get_model(config)
    model.to(device)
    optimizer = get_optimizer(config, model.parameters())
    return model, optimizer


def load_checkpoint_if_needed(model, optimizer, opt):
    """Load the checkpoint if resume is True."""
    start_epoch = 0

    if opt.resume:
        ckpt = load_checkpoint(opt.path_to_checkpoint)
        model.load_state_dict(ckpt["net"])
        start_epoch = ckpt["epoch"]
        optimizer.load_state_dict(ckpt["optim"])
        print("Last checkpoint restored")
    return model, optimizer, start_epoch

class EpochsDataStorage:
    """
    A class to store data for each epoch during training.

    Attributes:
        train_true (list): True labels for training data.
        train_pred (list): Predicted labels for training data.
        test_true (list): True labels for testing data.
        test_pred (list): Predicted labels for testing data.
        metrics_all_epochs (list): Metrics calculated at each epoch.

    Methods:
        reset_epoch_data: Reset the data for a new epoch.
        add_train_data: Add true and predicted labels for training data.
        add_test_data: Add true and predicted labels for testing data.
        add_metrics: Add metrics calculated at an epoch.
        get_train_data: Get concatenated true and predicted labels for training data.
        get_test_data: Get concatenated true and predicted labels for testing data.
        get_metrics: Get all the metrics calculated across epochs.

    """
    
    def __init__(self):
        """
        Initialize the EpochsDataStorage instance.
        """
        
        self.train_true = []
        self.train_pred = []

        self.test_true = []
        self.test_pred = []
        
        self.metrics_all_epochs = []

    def reset_epoch_data(self):
        """
        Reset the data to prepare for a new epoch.
        """
        
        self.train_true = []
        self.train_pred = []
        self.test_true = []
        self.test_pred = []

    def _process_predictions(self, predictions):
        """
        Process predicted labels by applying sigmoid and setting highest value to 1.

        Args:
            predictions (torch.Tensor): Predicted labels.

        Returns:
            np.ndarray: Processed predicted labels.
        """
        predictions = predictions.detach().cpu()
        processed_pred = torch.zeros_like(predictions)
        processed_pred[torch.arange(predictions.shape[0]), predictions.argmax(dim=1)] = 1
        return processed_pred.numpy()

    def add_train_data(self, true_labels, pred_labels):
        """
        Add true and predicted labels for training data.

        Args:
            true_labels (torch.Tensor): True labels.
            pred_labels (torch.Tensor): Predicted labels.
        """
        self.train_true.append(true_labels.detach().cpu().numpy())
        self.train_pred.append(self._process_predictions(pred_labels))
        
        

    def add_test_data(self, true_labels, pred_labels):
        """
        Add true and predicted labels for testing data.

        Args:
            true_labels (torch.Tensor): True labels.
            pred_labels (torch.Tensor): Predicted labels.
        """
        self.test_true.append(true_labels.detach().cpu().numpy())
        self.test_pred.append(self._process_predictions(pred_labels))
        
    def add_metrics(self, metrics):
        """
        Add metrics calculated at an epoch.
        """
        self.metrics_all_epochs.append(metrics)

    def get_train_data(self):
        """
        Get concatenated true and predicted labels for training data.
        """
        
        return np.concatenate(self.train_true), np.concatenate(self.train_pred)

    def get_test_data(self):
        """
        Get concatenated true and predicted labels for testing data.
        """
        
        return np.concatenate(self.test_true), np.concatenate(self.test_pred)
    
    def get_metrics(self):
        """
        Get all the metrics calculated across epochs.

        Returns:
            list: Metrics calculated at each epoch.
        """
        
        return self.metrics_all_epochs