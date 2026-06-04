"""
Raw TSV -> cleaned parquet pipeline.
Author: Eeshan Gupta
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pycountry
from dotenv import load_dotenv
from tqdm import tqdm


def _country_name(region: str | float | None) -> str | None:
    if pd.isna(region):
        return None
    region_str = str(region).strip()
    if not region_str:
        return None
    if len(region_str) == 2:
        match = pycountry.countries.get(alpha_2=region_str)
        return match.name if match else region_str
    if len(region_str) == 3:
        match = pycountry.countries.get(alpha_3=region_str)
        return match.name if match else region_str
    return region_str


def pre_process_title_akas(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.rename(
        columns={
            "titleId": "title_id",
            "isOriginalTitle": "is_original_title",
        }
    )
    chunk["is_original_title"] = (
        pd.to_numeric(chunk["is_original_title"], errors="coerce").fillna(0) > 0
    )
    chunk["country_name"] = chunk["region"].map(_country_name)
    return chunk


def pre_process_title_basic(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["isAdult"] = pd.to_numeric(chunk["isAdult"], errors="coerce").fillna(0) > 0
    return chunk


def pre_process_title_episode(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["seasonNumber"] = pd.to_numeric(chunk["seasonNumber"], errors="coerce")
    chunk["episodeNumber"] = pd.to_numeric(chunk["episodeNumber"], errors="coerce")
    return chunk


def pre_process_dataset(dataset_name: str, dataset: pd.DataFrame) -> pd.DataFrame:
    if "title.akas" in dataset_name:
        return pre_process_title_akas(dataset)
    if "title.basics" in dataset_name:
        return pre_process_title_basic(dataset)
    if "title.episode" in dataset_name:
        return pre_process_title_episode(dataset)
    return dataset


def clean(dataset_name: str, chunk: pd.DataFrame) -> pd.DataFrame:
    chunk = chunk.replace(r"\N", np.nan)
    return pre_process_dataset(dataset_name, chunk)


def clean_raw_files(
    raw_data_path: Path,
    clean_data_path: Path,
    chunk_size: int,
    datasets: Iterable[str] | None = None,
) -> list[dict[str, int | str]]:
    clean_data_path.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(raw_data_path.glob("*.tsv"))
    selected = set(datasets or [])
    stats: list[dict[str, int | str]] = []

    for file in tqdm(raw_files, desc="Processing Files"):
        dataset_name = file.stem
        if selected and dataset_name not in selected:
            continue
        rows = 0
        chunks = 0
        reader = pd.read_csv(file, sep="\t", chunksize=chunk_size, low_memory=False)
        for idx, chunk in enumerate(tqdm(reader, desc=f"Chunks:{dataset_name}")):
            output_file = clean_data_path / f"{dataset_name}.{idx:06d}.parquet"
            cleaned_chunk = clean(dataset_name, chunk)
            rows += len(cleaned_chunk)
            chunks += 1
            cleaned_chunk.to_parquet(
                output_file, engine="pyarrow", compression="snappy", index=False
            )
        stats.append({"dataset": dataset_name, "rows": rows, "chunks": chunks})
    return stats


def main() -> None:
    load_dotenv(".env.dev")
    parser = argparse.ArgumentParser(description="Clean IMDB raw TSV files to parquet.")
    parser.add_argument("--raw-dir", default="./data/raw")
    parser.add_argument("--clean-dir", default="./data/cleaned")
    parser.add_argument("--chunk-size", type=int, default=100000)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names like title.basics title.akas",
    )
    args = parser.parse_args()

    stats = clean_raw_files(
        raw_data_path=Path(args.raw_dir),
        clean_data_path=Path(args.clean_dir),
        chunk_size=args.chunk_size,
        datasets=args.datasets,
    )
    for item in stats:
        print(f"{item['dataset']}: rows={item['rows']} chunks={item['chunks']}")


if __name__ == "__main__":
    main()
