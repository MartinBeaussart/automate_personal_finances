import torch

def get_loss(config: dict) -> torch.nn.modules.loss._Loss:
    """
    Returns a PyTorch loss function based on the provided configuration.

    Args:
        config (dict): A dictionary containing the loss function name and any additional parameters.
            Expected keys:
                - "loss" (str): The name of the loss function to use.
                - "loss_pos_weight" (float, optional): The positive weight for BCEWithLogitsLoss.

    Returns:
        torch.nn.modules.loss._Loss: The selected PyTorch loss function.
    """

    # Mapping of loss names to their corresponding PyTorch functions
    loss_functions = {
        "NLLLoss": torch.nn.NLLLoss(),
        "BCELLOG": torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(config.get("loss_pos_weight", 1.0))),
        "MAE": torch.nn.L1Loss(),  # Mean Absolute Error
        "CrossEntropy": torch.nn.CrossEntropyLoss(),
        "SmoothL1Loss": torch.nn.SmoothL1Loss(),
        "MSE": torch.nn.MSELoss()  # Mean Squared Error
    }

    # Retrieve the loss function from the config and return it
    loss_name = config["loss"]
    if loss_name not in loss_functions:
        raise ValueError(f"Unsupported loss function: {loss_name}")
    
    return loss_functions[loss_name]