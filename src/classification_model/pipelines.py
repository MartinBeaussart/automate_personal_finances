from src.classification_model.utils import (
    load_checkpoint,
    setup_device,
    load_model_and_optimizer,
    load_checkpoint_if_needed,
)
from src.classification_model.dataset import load_data, get_input_data_loader
import argparse
from src.classification_model.loss import get_loss
from src.classification_model.training import training
from src.classification_model.predicting import (
    get_input,
    predicting,
    update_dataframe_with_category,
)

INPUTPATH = "./data/finance/transactions/"


def train_model(config: dict, opt: argparse.Namespace):
    """
    Train a classification model.

    Args:
        config: Configuration dictionary containing hyperparameters and settings.
        opt: Command line options or arguments.
    """

    device = setup_device()

    # Load the model and optimizer based on the configuration
    model, optimizer = load_model_and_optimizer(config, device)

    # Define the loss function for the classification task
    loss_fn = get_loss(config)

    # Load the training and testing data loaders
    train_loader, test_loader = load_data(config)

    # Load a checkpoint if available/wanted
    model, optimizer, start_epoch = load_checkpoint_if_needed(model, optimizer, opt)

    # Start the training process
    training(
        train_loader,
        test_loader,
        model,
        loss_fn,
        optimizer,
        start_epoch,
        config,
        device,
    )


def process_new_transactions(config: dict, opt: argparse.Namespace):
    """
    Process new transactions located in data/input/, run the classification model on it and update the dataframe with a column 'category'.

    Args:
        config: Configuration dictionary containing hyperparameters and settings.
        opt: Command line options or arguments.
    """

    # Get the input data from data/input
    input_data = get_input()
    input_loader = get_input_data_loader(input_data, config)

    # Load the best-performing model from a checkpoint
    ckpt = load_checkpoint("checkpoints/experiment_1.ckpt")
    model, _ = load_model_and_optimizer(config, "cpu")
    model.load_state_dict(ckpt["net"])
    model.eval()

    # Run the classification model on the input data and get the predicted categories
    categories_found = predicting(input_loader, model)

    # Update the dataframe
    input_data = update_dataframe_with_category(input_data, categories_found)

    # Save the file in the folder data/finance/trasnsaction
    input_data.write_csv(INPUTPATH + "transaction_M_Y.csv")
