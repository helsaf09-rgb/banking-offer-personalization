from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


MBD_MINI_DATASET_URL = "https://huggingface.co/datasets/ai-lab/MBD-mini"
MBD_MINI_TARGETS_URL = f"{MBD_MINI_DATASET_URL}/resolve/main/targets.tar.gz"
MBD_MINI_CLIENT_SPLIT_URL = f"{MBD_MINI_DATASET_URL}/resolve/main/client_split.tar.gz"
MBD_PAPER_URL = "https://doi.org/10.48550/arXiv.2409.17587"

PRODUCT_LABELS: dict[str, str] = {
    "target_1": "MBD product 1",
    "target_2": "MBD product 2",
    "target_3": "MBD product 3",
    "target_4": "MBD product 4",
}


def _download_file(url: str, output_path: Path, force_download: bool = False) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not force_download:
        return output_path
    urllib.request.urlretrieve(url, output_path)
    if not output_path.exists():
        raise FileNotFoundError(f"Download failed: {output_path} not found.")
    return output_path


def _extract_tar(archive_path: Path, expected_dir: Path, force_extract: bool = False) -> Path:
    if expected_dir.exists() and not force_extract:
        return expected_dir
    expected_dir.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r:gz") as tar:
        tar.extractall(expected_dir.parent)
    if not expected_dir.exists():
        raise FileNotFoundError(f"Archive extraction failed: {expected_dir} not found.")
    return expected_dir


def prepare_mbd_mini_targets(
    data_dir: Path,
    force_download: bool = False,
    force_extract: bool = False,
    include_client_split: bool = True,
) -> Path:
    """Download and extract lightweight MBD-mini target files.

    The full MBD-mini repository contains large transaction, geo, and dialog archives.
    This project only needs product propensity labels for a tractable real banking
    product-holdout validation, so the loader fetches targets and, optionally,
    client split metadata.
    """

    data_dir.mkdir(parents=True, exist_ok=True)
    targets_archive = _download_file(
        MBD_MINI_TARGETS_URL,
        data_dir / "targets.tar.gz",
        force_download=force_download,
    )
    targets_dir = _extract_tar(
        targets_archive,
        data_dir / "targets",
        force_extract=force_extract,
    )

    if include_client_split:
        split_archive = _download_file(
            MBD_MINI_CLIENT_SPLIT_URL,
            data_dir / "client_split.tar.gz",
            force_download=force_download,
        )
        _extract_tar(
            split_archive,
            data_dir / "client_split",
            force_extract=force_extract,
        )

    return targets_dir


def _read_partitioned_parquet(dataset_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(dataset_dir.rglob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_dir}.")

    frames: list[pd.DataFrame] = []
    for parquet_path in parquet_files:
        frame = pd.read_parquet(parquet_path)
        fold_part = next((part for part in parquet_path.parts if part.startswith("fold=")), None)
        if fold_part is not None and "fold" not in frame.columns:
            frame["fold"] = int(fold_part.split("=", maxsplit=1)[1])
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def _build_positive_product_events(targets_df: pd.DataFrame) -> pd.DataFrame:
    required_columns = {"client_id", "mon", "trans_count", "diff_trans_date", *PRODUCT_LABELS.keys()}
    missing_columns = required_columns.difference(targets_df.columns)
    if missing_columns:
        raise ValueError(f"MBD-mini targets are missing required columns: {sorted(missing_columns)}")

    events = targets_df.melt(
        id_vars=["client_id", "mon", "trans_count", "diff_trans_date", "fold"],
        value_vars=list(PRODUCT_LABELS),
        var_name="target_name",
        value_name="label",
    )
    events["label"] = pd.to_numeric(events["label"], errors="coerce").fillna(0).astype(int)
    events = events[events["label"] > 0].copy()
    events["user_id"] = events["client_id"].astype(str)
    events["offer_id"] = events["target_name"].map(lambda value: value.replace("target_", "mbd_product_"))
    events["offer_name"] = events["target_name"].map(PRODUCT_LABELS)
    events["timestamp"] = pd.to_datetime(events["mon"], errors="coerce")
    events["amount"] = pd.to_numeric(events["trans_count"], errors="coerce").fillna(0.0).astype(float)
    events["diff_trans_date"] = pd.to_numeric(events["diff_trans_date"], errors="coerce")
    events = events.dropna(subset=["timestamp"]).copy()
    return events.sort_values(["user_id", "timestamp", "offer_id"]).reset_index(drop=True)


def load_mbd_mini_product_interactions(
    targets_dir: Path,
    min_user_positive_events: int = 2,
    max_users: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    targets_df = _read_partitioned_parquet(targets_dir)
    raw_rows = int(len(targets_df))
    raw_clients = int(targets_df["client_id"].nunique())
    months = sorted(targets_df["mon"].dropna().astype(str).unique().tolist())

    events = _build_positive_product_events(targets_df)
    positive_clients_before_filtering = int(events["user_id"].nunique())
    positive_events_before_filtering = int(len(events))

    user_event_counts = events.groupby("user_id").size()
    keep_users = user_event_counts[user_event_counts >= int(min_user_positive_events)].index
    events = events[events["user_id"].isin(keep_users)].copy()

    if max_users is not None and max_users > 0 and events["user_id"].nunique() > max_users:
        keep_users = (
            events.groupby("user_id")
            .size()
            .sort_values(ascending=False)
            .head(max_users)
            .index
        )
        events = events[events["user_id"].isin(keep_users)].copy()

    interactions = events[
        [
            "user_id",
            "offer_id",
            "timestamp",
            "amount",
            "diff_trans_date",
            "fold",
            "target_name",
            "label",
        ]
    ].copy()
    interactions = interactions.sort_values(["user_id", "timestamp", "offer_id"]).reset_index(drop=True)

    offers = (
        events.groupby(["offer_id", "offer_name", "target_name"], as_index=False)
        .agg(
            buyer_count=("user_id", "nunique"),
            event_count=("user_id", "size"),
            first_month=("timestamp", "min"),
            last_month=("timestamp", "max"),
        )
        .sort_values(["buyer_count", "event_count", "offer_id"], ascending=[False, False, True])
        .reset_index(drop=True)
    )

    product_positive_counts = {
        product_id: int(count)
        for product_id, count in interactions.groupby("offer_id").size().sort_index().items()
    }
    stats: dict[str, Any] = {
        "dataset_name": "MBD-mini targets",
        "dataset_origin": "Sber AI Lab / ai-lab MBD-mini",
        "source_url": MBD_MINI_DATASET_URL,
        "paper_url": MBD_PAPER_URL,
        "raw_target_rows": raw_rows,
        "raw_clients": raw_clients,
        "months": months,
        "min_user_positive_events": int(min_user_positive_events),
        "max_users": None if max_users is None else int(max_users),
        "positive_clients_before_filtering": positive_clients_before_filtering,
        "positive_events_before_filtering": positive_events_before_filtering,
        "n_users": int(interactions["user_id"].nunique()),
        "n_items": int(interactions["offer_id"].nunique()),
        "n_positive_interactions": int(len(interactions)),
        "timestamp_min": interactions["timestamp"].min().isoformat(),
        "timestamp_max": interactions["timestamp"].max().isoformat(),
        "product_positive_counts": product_positive_counts,
    }
    return interactions, offers, stats
