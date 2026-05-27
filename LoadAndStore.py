# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Loading, Cleaning and Storing
# Load the raw data in chunks, cleaning the chunks and then storing them for downstream usage.
# #### Author: [Eeshan Gupta](mailto:eeshan.gupta@hotmail.com)

# %% [markdown]
# ## Imports and Setup

# %%
import gc
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
from dotenv import load_dotenv
from tqdm import tqdm

# %%
load_dotenv("./.env.dev")
pd.set_option("display.max_rows", 500)

# %%
RAW_DATA_PATH = os.getenv("RAW_DATA_DIR", "./data/raw/")
RAW_DATA_PATH = Path(RAW_DATA_PATH)

CLEAN_DATA_PATH = os.getenv("CLEANED_DATA_DIR", "./data/cleaned/")
CLEAN_DATA_PATH = Path(CLEAN_DATA_PATH)

CHUNK_SIZE = os.getenv("CHUNK_SIZE", "100")
CHUNK_SIZE = int(CHUNK_SIZE)

CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)

# %%
RAW_FILES = list(RAW_DATA_PATH.glob("*.tsv"))

# %% [markdown]
# ## Cleaning the data
# Cleaning up each dataset and storing them as parquet for downstream processing.


# %%
def pre_process_title_akas(chunk_: pd.DataFrame) -> pd.DataFrame:
    """
    Pre-processes a DataFrame containing title and region information.

    The function processes a DataFrame, specifically targeting columns related to
    title and region data. The 'isOriginalTitle' column is transformed into a boolean
    indicator determining if the title is original. Additionally, the 'region' column
    is used to derive the full country name, which is appended to a new column
    'country_name'.

    Parameters
    ----------
    chunk_ : pd.DataFrame
        DataFrame containing title and region information, including columns
        'isOriginalTitle' and 'region'.

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with 'is_original_title' and 'country_name' columns added.
    """

    def get_country_name(region: str) -> str:
        try:
            assert not pd.isnull(region)
            assert len(region) <= 3
            if len(region) == 2:
                return pycountry.countries.get(alpha_2=region).name
            elif len(region) == 3:
                return pycountry.countries.get(alpha_3=region).name
        except AssertionError:
            return region
        except AttributeError:
            return region

    chunk_["is_original_title"] = chunk_["isOriginalTitle"].transform(lambda x: x > 0)
    chunk_["country_name"] = chunk_["region"].transform(get_country_name)
    return chunk_


# %%
def pre_process_title_basic(chunk_: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the `title.basics` dataset by transforming the `isAdult` column.

    The function converts the `isAdult` column in the provided chunk of data into
    a boolean format, where the values are set to `True` if they are greater than
    0, otherwise `False`.

    Parameters
    ----------
    chunk_ : pd.DataFrame
        A DataFrame containing a chunk of the `title.basics` dataset.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the `isAdult` column transformed to boolean values.
    """
    chunk_["isAdult"] = chunk_["isAdult"].transform(lambda x: x > 0)
    return chunk_


# %%
def pre_process_title_episode(chunk_: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the 'seasonNumber' and 'episodeNumber' columns in the given
    DataFrame to a numeric type with a reduced memory footprint. Both
    columns are downcast to the float type to optimize memory usage.

    Parameters
    ----------
    chunk_ : pd.DataFrame
        The DataFrame containing the 'seasonNumber' and 'episodeNumber'
        columns to be processed.

    Returns
    -------
    pd.DataFrame
        The modified DataFrame with 'seasonNumber' and 'episodeNumber'
        columns converted to a numeric type.
    """
    chunk_["seasonNumber"] = pd.to_numeric(chunk_["seasonNumber"], downcast="float")
    chunk_["episodeNumber"] = pd.to_numeric(chunk_["episodeNumber"], downcast="float")
    return chunk_


# %%
def pre_process_dataset(dataset_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Processes datasets based on their specific types. Depending on the name of the
    dataset provided, this function applies the appropriate preprocessing routine.
    It aims to ensure each dataset is correctly formatted or refined for subsequent
    analysis.

    Parameters
    ----------
    dataset_name : str
        The name of the dataset, used to determine the applicable preprocessing
        method.
    dataset : pd.DataFrame
        The dataset to be processed.

    Returns
    -------
    pd.DataFrame
        The processed dataset, adjusted according to its type.
    """
    if "title.akas" in dataset_name:
        return pre_process_title_akas(dataset)

    if "title.basics" in dataset_name:
        return pre_process_title_basic(dataset)

    if "title.ratings" in dataset_name:
        return dataset

    if "name.basics" in dataset_name:
        return dataset

    if "title.crew" in dataset_name:
        return dataset

    if "title.principals" in dataset_name:
        return dataset

    if "title.episode" in dataset_name:
        return pre_process_title_episode(dataset)

    return dataset


# %%
def clean(file_name: str, chunk_: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the specified dataset by replacing specific placeholders with NaN
    values and processes it further.

    This function targets the given DataFrame, replacing all instances of
    the placeholder "\\N" with NaN, indicating missing values. Following this
    operation, it calls another function to perform additional preprocessing
    on the dataset. The function ultimately returns the cleaned DataFrame,
    ready for further data analysis tasks.

    Parameters
    ----------
    file_name : str
        The name of the file associated with the dataset to be cleaned. This
        is used during the preprocessing step.
    chunk_ : pd.DataFrame
        The DataFrame that contains a chunk of the dataset to be cleaned.
        This DataFrame is modified directly by replacing occurrences of the
        placeholder "\\N" with NaN values.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the same data as the input, but with specified
        placeholders replaced by NaN values and preprocessed as necessary.
    """
    chunk_ = chunk_.replace(r"\N", np.nan)
    chunk_ = pre_process_dataset(file_name, chunk_)
    return chunk_


# %%
cleaning_process = False
if cleaning_process:
    for file in tqdm(RAW_FILES, desc="Processing Files"):
        reader = pd.read_csv(
            file,
            sep="\t",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
        for idx, chunk in enumerate(tqdm(reader, desc="Processing Chunks")):
            output_file_name = Path(file).with_suffix(f".{idx:06d}.parquet")
            output_file_name = CLEAN_DATA_PATH / output_file_name.name
            cleaned_chunk = clean(output_file_name.name, chunk)
            cleaned_chunk.to_parquet(
                output_file_name, engine="pyarrow", compression="snappy", index=False
            )
            del chunk
            del cleaned_chunk
            gc.collect()
else:
    print("Cleaned Parquet files are available")

# %%
table_names = [
    str(file.name).replace(".tsv", "").replace(".", "_").upper()
    for file in RAW_DATA_PATH.glob("*.tsv")
]

# %%
for table in table_names:
    if table in {}:
        continue
    cleaned_pattern = table.lower().replace("_", ".")
    files = CLEAN_DATA_PATH.glob(f"*{cleaned_pattern}.*.*")
    temp_file = list(files)[0]
    df = pd.read_parquet(temp_file)
    break
