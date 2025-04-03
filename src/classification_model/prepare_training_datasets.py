import os
import random
import argparse
from datetime import datetime, timedelta

import polars as pl

RAWTRAINPATH = "./data/raw_data/train/"
RAWTESTPATH = "./data/raw_data/test/"
OUTPUT_PATH = "./data/clean_data/"
TRANSACTIONS_PER_FILE = 200


# TODO save the category with the id in a file
def create_dataset(config: dict, opt: argparse.Namespace):
    """
    Creates and processes the training and test datasets.
    """

    if config["use_synthetic_data"]:
        df = generate_transactions(TRANSACTIONS_PER_FILE)
        file_path = os.path.join(RAWTRAINPATH, f"swiss_bank_transactions.csv")
        df.write_csv(file_path)

    # Get all raw train data
    try:
        train_data = load_and_concatenate_data(RAWTRAINPATH)
    except Exception as e:
        print(f"Error reading training data files: {e}")
        return

    # Encode categories in the training dataset
    train_data, cat_encoding = encode_categories(train_data)

    # Get all raw test data
    try:
        test_data = load_and_concatenate_data(RAWTESTPATH)
    except Exception as e:
        print(f"Error reading test data files: {e}")
        return

    # Encode categories in the test dataset using the same mapping as training
    test_data = test_data.join(cat_encoding, on="Category")
    
    size_list = train_data.select("Category_encoded").max().item() + 1

     # Create one-hot encoded labels for both datasets
    train_data = create_one_hot_labels(train_data, size_list)
    test_data = create_one_hot_labels(test_data, size_list)

    train_data.write_parquet(os.path.join(OUTPUT_PATH, "train.parquet"))
    test_data.write_parquet(os.path.join(OUTPUT_PATH, "test.parquet"))


def generate_transactions(num_transactions: int) -> pl.DataFrame:
    """
    Generates synthetic transaction data.
    """
    
    category_mapping = {
        "Rent": [
            "Mietzahlung",
            "Wohnungskaution",
            "Hypothekenzahlung",
            "Renovierungskosten",
            "Paiement de l'hypothèque",
            "Nebenkosten",
            "logement",
            "Loyer",
            "Frais de rénovation",
            "Commune de Lausanne",
            "Commune de Geneve",
            "canton VD (ECA)",
            "Etabl. ass. contre l'incendie",
        ],
        "Shopping": [
            "IKEA",
            "Mediamarkt",
            "Amazon",
            "Globus",
            "Decathlon",
            "Manor",
            "Zalando",
            "H&M",
            "Fnac",
            "payot",
            "Media Markt",
            "PKZ",
            "Zara",
            "Conforama",
            "Dosenbach",
            "Aeschbach",
            "ANOUK",
            "Apple",
            "Blue Tomato",
            "Etam",
            "GIFI",
        ],
        "Salary": [
            "Lohnüberweisung",
            "Bonuszahlung",
            "Nebeneinkommen",
            "Salaire",
            "Paiement de bonus",
            "Revenu",
        ],
        "Outdoor food": [
            "Restaurant Zürich",
            "Café Latte",
            "Bäckerei Müller",
            "Takeaway Pizza",
            "Pizzeria Rechnung",
            "Barzahlung Kneipe",
            "Brasserie la Source",
            "Freilager La Cucina Colaianni",
            "La Miranda Gourmet Stübli",
            "Restaurant",
            "Mikuriya",
            "Bistro",
            "La Table Du Lausanne Palace",
            "Brasserie Saint Laurent",
            "Ristorante Amici",
            "Le Café de Grancy",
            "Café Romand",
            "Café du Vieil Ouchy",
        ],
        "Transport": [
            "SBB",
            "Tankstelle AG",
            "Parkhaus Gebühr",
            "Taxi",
            "ÖV Monatskarte",
            "Carsharing Gebühr",
            "Station-service",
            "Parking",
            "transports publics",
            "autopartage",
            "CFF",
            "Train",
            "Bus",
            "TPG",
            "transport",
            "metro",
        ],
        "Grocery": [
            "Migros",
            "Coop",
            "Denner",
            "Aldi",
            "Lidl Markt",
            "Auchan",
            "Intermarché",
            "costco",
            "wal-mart",
            "walmart",
        ],
        "Leisure": [
            "Kino Pathé",
            "Cinema",
            "Gaumont pathé",
            "Theater",
            "TheatreEscape Room",
            "Museum",
            "Musée",
            "Sport",
            "Football",
            "Circus",
            "cirque",
            "disney",
            "Concert",
            "Club",
            "Hotel",
            "Flugtickets",
            "Ferienwohnung Miete",
            "Reiseversicherung",
            "Billets d'avion",
            "Location de vacances",
            "Assurance voyage",
            "Motel",
        ],
        "Mobile": [
            "Facture de téléphone Swisscom",
            "Facture Salt",
        ],
        "Health": [
            "Arztpraxis Rechnung",
            "Apotheke Bezahlung",
            "Frisör Haarschnitt",
            "Yoga",
            "Massagestudio",
            "Cabinet médical",
            "pharmacie",
            "Coiffeur",
            "massage",
            "Bain thermaux",
            "Docteur",
            "Hospital",
            "hopital",
            "CHUV",
            "Dentist",
        ],
        "Assurance": [
            "Autoversicherung",
            "Krankenkasse",
            "Hausratversicherung",
            "Lebensversicherung",
            "Assurance automobile",
            "assurance maladie",
            "Assurance ménage",
            "Assurance",
            "Assurance vie",
            "Helsana",
            "Assura",
            "Concordia",
            "Group Mutuel",
            "Caisse Maladie",
        ],
        "Administrative": [
            "Kontoführungsgebühr",
            "Kreditkartengebühr",
            "Frais de tenue de compte",
            "Frais de carte de crédit",
        ],
    }

    data = []
    for _ in range(num_transactions):
        date = datetime.now() - timedelta(days=random.randint(0, 365))
        category = random.choice(list(category_mapping.keys()))
        description = random.choice(category_mapping[category])
        
        data.append([date.strftime("%Y-%m-%d"), description, category])

    return pl.DataFrame(data, schema={"Date": pl.Utf8, "Description": pl.Utf8, "Category": pl.Utf8})


def load_and_concatenate_data(directory: str) -> pl.DataFrame:
    """
    Loads and concatenates all CSV files in a given directory.
    """
    
    list_csv = [f for f in os.listdir(directory) if f.endswith('.csv')]
    data_frames = [pl.read_csv(os.path.join(directory, file_name)) for file_name in list_csv]
    return pl.concat(data_frames)

def encode_categories(df: pl.DataFrame):
    """
    Encodes categories in the DataFrame to integer codes.
    """
    
    df = df.with_columns(
        pl.col("Category").rank("dense").cast(pl.Int64).alias("Category_encoded")
    )
    cat_encoding = df.select(["Category", "Category_encoded"]).unique()
    return df, cat_encoding

def create_one_hot_labels(df: pl.DataFrame, size_list: int) -> pl.DataFrame:
    """
    Creates one-hot encoded labels based on the 'Category_encoded' column.
    """

    def create_label(position):
        zero_list = [0] * size_list
        zero_list[position] = 1
        return zero_list

    return df.with_columns(
        pl.col("Category_encoded")
        .map_elements(create_label)
        .alias("label"),
    )
