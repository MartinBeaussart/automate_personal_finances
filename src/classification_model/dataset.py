import numpy as np
import tables
import polars as pl
import torch
from torch.utils import data
from torchvision import transforms
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset

TOKENIZER = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def load_data(config: dict) -> tuple:
    """
    Create train and test data loaders based on the provided configuration.

    Args:
        config (dict): Dictionary containing batch size, number of workers, etc.

    Returns:
        tuple: Train and test data loaders
    """
    # Get train and test datasets
    train_dataset = get_train_data(config)
    test_dataset = get_test_data(config)

    # Create data loaders for train and test sets
    train_data_loader = data.DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"]
    )

    test_data_loader = data.DataLoader(
        test_dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True, num_workers=config["num_workers"]
    )

    return train_data_loader, test_data_loader

def get_input_data_loader(_data, config):
    dataset = MultiLabelDataset(_data, TOKENIZER, 128)
    
    data_loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, drop_last=True, num_workers=config["num_workers"]
    )
    
    return data_loader
    


def train_transforms() -> transforms.Compose:
    """
    Define a composition of transformations for training.

    Returns:
        transforms.Compose: A composition of ToTensor and RandomAffine transformations.
    """
    transform = transforms.Compose(
        [
            # Convert to tensor
            transforms.ToTensor(),
            # Apply random affine transformation with 0 degrees rotation and up to 2% translation
            transforms.RandomAffine(degrees=0, translate=(0.0, 0.02)),
        ]
    )
    return transform


def test_transforms() -> transforms.Compose:
    """
    Define a composition of transformations for testing.

    Returns:
        transforms.Compose: A composition containing only ToTensor transformation.
    """
    transform = transforms.Compose(
        [
            # Convert to tensor
            transforms.ToTensor(),
        ]
    )
    return transform



def get_test_data(config: dict):
    """
    Get test dataset based on the provided configuration.

    Args:
        config (dict): Dictionary containing batch size, number of workers, etc.

    Returns:
        MultiLabelDataset: Test dataset
    """

    return MultiLabelDataset(pl.read_parquet(config['test_file']), TOKENIZER, 128)


def get_train_data(config: dict):
    """
    Get train dataset based on the provided configuration.

    Args:
        config (dict): Dictionary containing batch size, number of workers, etc.

    Returns:
        MultiLabelDataset: Train dataset
    """

    return MultiLabelDataset(pl.read_parquet(config['train_file']), TOKENIZER, 128)

class MultiLabelDataset(Dataset):
    """
    A PyTorch Dataset class for handling multi-label text classification.

    Args:
        dataframe (pl.DataFrame): Polars DataFrame containing data.
        tokenizer: Tokenizer instance (e.g., DistilBertTokenizer).
        max_len (int): Maximum length of the input sequence.
    """

    def __init__(self, dataframe: pl.DataFrame, tokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = self.data.get_column('Description').to_list()
        self.targets = self.data.get_column('label').to_list()
        self.max_len = max_len

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items.
        """
        return len(self.text)

    def __getitem__(self, index: int) -> dict:
        """
        Get an item from the dataset by its index.

        Args:
            index (int): Index of the item.

        Returns:
            dict: Dictionary containing input IDs, attention mask, and target.
        """
        # Get text and target at the specified index
        text = str(self.text[index])
        text = " ".join(text.split())

        # Tokenize the text
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        
        # Get input IDs, attention mask, and target
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

    
class Dataset_H5(object):
    """
    A class for handling HDF5 datasets.

    Args:
        fname (str): Name of the HDF5 file.
        transform: Transformation function to apply to the data.
    """

    def __init__(self, fname: str, transform=None):
        self.fname = fname
        self.transform = transform

        with tables.open_file(self.fname, 'r') as db:
            # Load patient IDs, labels, and original time frames from HDF5 file
            self.patid = db.root.patid[:].astype(np.int64)
            self.label = db.root.label[:].astype(np.float32)
            self.original_time_frame = db.root.original_time_frame[:].astype(np.uint8)

        # Get the number of items in the dataset
        self.nitems = len(self.patid)

    def __getitem__(self, index: int) -> tuple:
        """
        Get an item from the dataset by its index.

        Args:
            index (int): Index of the item.

        Returns:
            tuple: Tuple containing features, label, patient ID, and original time frame.
        """
        # Load features from HDF5 file
        with tables.open_file(self.fname, 'r') as db:
            features = db.root.features[index, :, :].astype(np.float32)

        # Get label, patient ID, and original time frame at the specified index
        label = self.label[index]
        patid = self.patid[index]
        original_time_frame = self.original_time_frame[index]

        # Apply transformation if provided
        if self.transform is not None:
            features = self.transform(features)

        return features, label, patid, original_time_frame

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Number of items.
        """
        return self.nitems