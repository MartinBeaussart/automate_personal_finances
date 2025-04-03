from torch.optim import SGD, Adam


def get_optimizer(config: dict, model_parameters: list) -> object:
    """
    Returns the optimizer based on the config file.

    Args:
        config (dict): The configuration dictionary containing the optimizer type and parameters.
        model_parameters (list): The model's trainable parameters.

    Returns:
        object: The initialized optimizer instance.

    Raises:
        NotImplementedError: If the specified optimizer is not supported.
    """
    # Get the optimizer type from the config
    optimizer_type = config["optimizer"]

    # Define a dictionary mapping optimizer types to their respective classes and accepted parameters
    optimizers = {
        "SGD": {"class": SGD, "params": ["lr", "momentum", "weight_decay"]},
        "Adam": {
            "class": Adam,
            "params": ["lr", "betas", "eps", "weight_decay", "amsgrad"],
        },
    }

    # Check if the optimizer type is supported
    if optimizer_type not in optimizers:
        raise NotImplementedError(
            f"The following optimizer is not supported: {optimizer_type}"
        )

    # Get the accepted parameters for the specified optimizer type
    accepted_params = optimizers[optimizer_type]["params"]

    # Filter the given parameters to only include the accepted ones
    optimizer_params = _get_params(accepted_params, config.get("optimizer_params", {}))

    # Initialize and return the optimizer instance
    return optimizers[optimizer_type]["class"](model_parameters, **optimizer_params)


def _get_params(accepted_params: list, given_params: dict) -> dict:
    """
    Filters the given parameters to only include the accepted ones.

    Args:
        accepted_params (list): The list of accepted parameter names.
        given_params (dict): The dictionary containing the given parameters.

    Returns:
        dict: A dictionary containing only the accepted parameters and their values.
    """
    # Use a dictionary comprehension to filter the given parameters
    return {
        param: float(given_params[param])
        for param in accepted_params
        if param in given_params
    }
