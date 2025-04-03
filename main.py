import argparse

from src.classification_model.utils import load_config_and_set_seed

from src.classification_model.prepare_training_datasets import create_dataset
from src.classification_model.pipelines import train_model, process_new_transactions


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a network.")
    parser.add_argument(
        "--name_config",
        type=str,
        default="default.yml",
        help="Name of the config file (default: 'default.yml')",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="If set, resumes training from checkpoint.",
    )
    parser.add_argument(
        "--path_to_checkpoint",
        type=str,
        default="",
        help="Path to checkpoint to resume training.",
    )
    return parser.parse_args()


def main():
    opt = parse_args()
    config = load_config_and_set_seed(f"./configs/{opt.name_config}")

    pipelines = {
        "create_dataset": create_dataset,
        "train_model": train_model,
        "process_new_transactions": process_new_transactions,
    }

    for pipeline_to_run in config["run_pipelines"]:
        pipelines[pipeline_to_run](config, opt)


if __name__ == "__main__":
    main()
