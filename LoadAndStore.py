# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.2
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

import pycountry
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# %%
load_dotenv("./.env.dev")
pd.set_option('display.max_rows', 500)

# %%
RAW_DATA_PATH = os.getenv("RAW_DATA_DIR")
CLEAN_DATA_PATH = os.getenv("CLEANED_DATA_DIR")
CLEAN_DATA_PATH = Path(CLEAN_DATA_PATH)
CHUNK_SIZE = os.getenv("CHUNK_SIZE", "100")
CHUNK_SIZE = int(CHUNK_SIZE)

CLEAN_DATA_PATH.mkdir(parents=True, exist_ok=True)

# %%
RAW_FILES = [
    os.path.join(RAW_DATA_PATH, file)
    for file in os.listdir(RAW_DATA_PATH)
    if file.endswith(".tsv")
]


# %% [markdown]
# ## Cleaning the data
# Cleaning up each dataset and storing them as parquet for downstream processing.

# %%
def clean(file_name: str, chunk_: pd.DataFrame) -> pd.DataFrame:
    chunk_ = chunk_.replace(r'\N', np.nan)
    # if "akas" in file_name:
    #     pass
    return chunk_


# %%
cleaning_process = False
if cleaning_process:
    for file in tqdm(RAW_FILES, desc="Processing Files"):
        print(file)
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

# %% [markdown]
# ## Pre-processing Data
# Loading `.parquet`s and pre-processing and adding context to the data

# %%
dev_stop = False
for file_name in os.listdir(RAW_DATA_PATH):
    dataset_name = ".".join(file_name.split('.')[:-1])
    parquet_files = sorted(list(CLEAN_DATA_PATH.glob(f"**/{dataset_name}.*.*")))
    for file_path in parquet_files:
        df = pd.read_parquet(file_path)
        dev_stop = True
        break
    if dev_stop:
        break


# %%
def pre_process_title_akas(chunk_: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the chunks of title.akas dataset.
    :param chunk_: pd.DataFrame, A chunk of title.akas dataset containing 100_000 datapoints.
    :return: pd.DataFrame, processed chunk.
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


    chunk_['is_original_title'] = chunk_['isOriginalTitle'].transform(lambda x: x >0)
    chunk_['country_name'] = chunk_['region'].transform(get_country_name)
    return chunk_

# %%
df = pre_process_title_akas(df)

# %%
df.head(20)

# %%
