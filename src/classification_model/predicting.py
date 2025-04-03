import os
import polars as pl
import torch
from tqdm import tqdm

INPUT_PATH = "./data/input/"
OUTPUT_PATH = "./data/clean_data/"

# TODO handle and clean data from different sources


def get_input() -> pl.DataFrame:
    """
    Retrieves the input data from various CSV files in the INPUT_PATH directory.

    Returns:
        A concatenated Polars DataFrame containing all the input data.
    """
    list_csv = os.listdir(INPUT_PATH)

    # Concatenate all the CSV files into a single DataFrame
    data = pl.concat([pl.read_csv(INPUT_PATH + file_name) for file_name in list_csv])

    # Initialize a new column 'label' with default value 0
    data = data.with_columns(label=0)

    return data


def predicting(
    input_loader: torch.utils.data.DataLoader, model: torch.nn.Module
) -> list:
    """
    Makes predictions on the input data using the provided model.

    Args:
        input_loader: A PyTorch DataLoader containing the input data.
        model: A trained PyTorch model used for making predictions.

    Returns:
        A list of predicted labels.
    """

    outputs = []

    # Disable gradient computation for faster inference
    with torch.no_grad():
        # Iterate over the testing dataset
        for _, data in tqdm(enumerate(input_loader), total=len(input_loader)):
            # data preparation
            ids = data["ids"].to("cpu", dtype=torch.long)
            mask = data["mask"].to("cpu", dtype=torch.long)

            out = model(ids, mask)

            # Get the predicted label by taking the argmax of the output tensor
            out = out.argmax(dim=1).item()

            outputs.append(out)

    return outputs


def update_dataframe_with_category(data: pl.DataFrame, category: list) -> pl.DataFrame:
    """
    Updates a DataFrame by adding a the column 'category'.

    Args:
        data: The input DataFrame to be updated.
        category: A list of category to be added as a new column.

    Returns:
        The updated DataFrame with the new column.
    """

    # Transform the list of category labels into a DataFrame
    category = pl.DataFrame(data={"label": category})

    # add the column with the predicted label
    data = pl.concat([data.drop("label"), category], how="horizontal")

    # TODO need to be a file only for that
    # Load a mapping of Category_encoded to Category from a Parquet file 
    train_data = pl.read_parquet(OUTPUT_PATH + "train.parquet")
    category_mapping = (
        train_data.select(["Category_encoded", "Category"])
        .unique()
        .sort("Category_encoded")
    )

    # Join the updated DataFrame with the category mapping using the 'label' column
    data = data.join(category_mapping, left_on="label", right_on="Category_encoded")

    return data
