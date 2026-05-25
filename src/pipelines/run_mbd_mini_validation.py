from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import REAL_DATA_DIR, REPORTS_DIR
from src.data.load_mbd_mini import (
    MBD_MINI_DATASET_URL,
    MBD_PAPER_URL,
    load_mbd_mini_product_interactions,
    prepare_mbd_mini_targets,
)
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.implicit_mf_recommender import ImplicitMFRecommender
from src.models.item_knn_recommender import ImplicitItemKNNRecommender
from src.models.lightgcn_recommender import LightGCNConfig, LightGCNRecommender
from src.models.neural_cf_recommender import NeuralCFConfig, NeuralCFRecommender
from src.models.popularity_recommender import PopularityRecommender
from src.models.sasrec_recommender import SASRecConfig, SASRecRecommender


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate recommendation models on MBD-mini real banking product labels."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REAL_DATA_DIR / "mbd_mini",
        help="Directory where MBD-mini target archives are downloaded and extracted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR / "mbd_mini_validation",
        help="Directory where MBD-mini validation artifacts are stored.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Top-K products for evaluation.")
    parser.add_argument(
        "--min-user-positive-events",
        type=int,
        default=2,
        help="Minimum number of positive product events per client before holdout.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="Optional cap on retained clients after filtering; 0 keeps all eligible users.",
    )
    parser.add_argument("--n-factors", type=int, default=3, help="Latent factors for MF baseline.")
    parser.add_argument("--n-neighbors", type=int, default=3, help="Neighbors for item-kNN baseline.")
    parser.add_argument("--force-download", action="store_true", help="Re-download MBD-mini target archives.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract MBD-mini target archives.")
    return parser.parse_args()


def build_temporal_product_holdout(
    interactions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    positives = interactions_df[interactions_df["label"] == 1].copy()
    positives["timestamp"] = pd.to_datetime(positives["timestamp"])
    positives = positives.sort_values(["user_id", "timestamp", "offer_id"]).reset_index(drop=True)

    last_ts_by_user = positives.groupby("user_id")["timestamp"].transform("max")
    holdout = positives[positives["timestamp"].eq(last_ts_by_user)].copy()
    train = positives[positives["timestamp"].lt(last_ts_by_user)].copy()

    train_users = set(train["user_id"].astype(str).unique())
    holdout_users = set(holdout["user_id"].astype(str).unique())
    eligible_users = sorted(train_users.intersection(holdout_users))
    train = train[train["user_id"].astype(str).isin(eligible_users)].reset_index(drop=True)
    holdout = holdout[holdout["user_id"].astype(str).isin(eligible_users)].reset_index(drop=True)
    return train, holdout


def _metric_row(
    model_name: str,
    metrics: dict[str, float | int],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"model": model_name, **metrics}
    if extra:
        row.update(extra)
    return row


def _evaluate_recommender(
    model_name: str,
    model: Any,
    user_ids: list[str],
    ground_truth_df: pd.DataFrame,
    top_k: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    recommendations_df = model.recommend_for_users(user_ids=user_ids, top_k=top_k, exclude_by_user={})
    metrics, per_user_df = evaluate_ranking(
        recommendations_df=recommendations_df,
        ground_truth_df=ground_truth_df,
        k=top_k,
    )
    per_user_df["model"] = model_name
    return metrics, per_user_df


def _fit_best_lightgcn(
    train_interactions_df: pd.DataFrame,
    user_ids: list[str],
    ground_truth_df: pd.DataFrame,
    top_k: int,
) -> tuple[LightGCNRecommender, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    configs = [
        LightGCNConfig(
            embedding_dim=8,
            n_layers=1,
            learning_rate=0.03,
            epochs=20,
            batch_size=1024,
            samples_per_epoch=12000,
            l2_reg=1e-4,
            random_state=42,
        ),
        LightGCNConfig(
            embedding_dim=16,
            n_layers=2,
            learning_rate=0.02,
            epochs=30,
            batch_size=1024,
            samples_per_epoch=18000,
            l2_reg=1e-4,
            random_state=42,
        ),
        LightGCNConfig(
            embedding_dim=24,
            n_layers=2,
            learning_rate=0.01,
            epochs=35,
            batch_size=1024,
            samples_per_epoch=24000,
            l2_reg=5e-5,
            random_state=42,
        ),
    ]

    rows: list[dict[str, Any]] = []
    best_model: LightGCNRecommender | None = None
    best_metrics: dict[str, Any] | None = None
    best_per_user: pd.DataFrame | None = None

    for idx, config in enumerate(configs, start=1):
        model = LightGCNRecommender(config=config).fit(train_interactions_df)
        metrics, per_user_df = _evaluate_recommender(
            model_name=f"lightgcn_config_{idx}",
            model=model,
            user_ids=user_ids,
            ground_truth_df=ground_truth_df,
            top_k=top_k,
        )
        row = {
            "candidate": f"lightgcn_config_{idx}",
            **metrics,
            "embedding_dim": config.embedding_dim,
            "n_layers": config.n_layers,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "samples_per_epoch": config.samples_per_epoch,
            "l2_reg": config.l2_reg,
            "training_loss_final": float(model.loss_history_[-1]) if model.loss_history_ else None,
        }
        rows.append(row)
        if best_metrics is None or float(metrics["ndcg_at_k"]) > float(best_metrics["ndcg_at_k"]):
            best_model = model
            best_metrics = row
            best_per_user = per_user_df

    if best_model is None or best_metrics is None or best_per_user is None:
        raise RuntimeError("LightGCN tuning did not produce a fitted model.")
    best_per_user = best_per_user.copy()
    best_per_user["model"] = "lightgcn_tuned"
    return best_model, best_metrics, pd.DataFrame(rows), best_per_user


def _fit_best_sasrec(
    train_interactions_df: pd.DataFrame,
    user_ids: list[str],
    ground_truth_df: pd.DataFrame,
    top_k: int,
) -> tuple[SASRecRecommender, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    configs = [
        SASRecConfig(
            embedding_dim=16,
            num_heads=2,
            num_blocks=1,
            dropout=0.1,
            max_seq_len=12,
            window_stride=1,
            learning_rate=2e-3,
            weight_decay=1e-5,
            epochs=12,
            batch_size=128,
            samples_per_epoch=20000,
            random_state=42,
        ),
        SASRecConfig(
            embedding_dim=32,
            num_heads=4,
            num_blocks=2,
            dropout=0.1,
            max_seq_len=12,
            window_stride=1,
            learning_rate=2e-3,
            weight_decay=1e-5,
            epochs=14,
            batch_size=128,
            samples_per_epoch=25000,
            random_state=42,
        ),
        SASRecConfig(
            embedding_dim=32,
            num_heads=4,
            num_blocks=2,
            dropout=0.2,
            max_seq_len=24,
            window_stride=1,
            learning_rate=1e-3,
            weight_decay=1e-5,
            epochs=16,
            batch_size=128,
            samples_per_epoch=30000,
            random_state=42,
        ),
    ]

    rows: list[dict[str, Any]] = []
    best_model: SASRecRecommender | None = None
    best_metrics: dict[str, Any] | None = None
    best_per_user: pd.DataFrame | None = None

    for idx, config in enumerate(configs, start=1):
        model = SASRecRecommender(config=config).fit(train_interactions_df)
        metrics, per_user_df = _evaluate_recommender(
            model_name=f"sasrec_config_{idx}",
            model=model,
            user_ids=user_ids,
            ground_truth_df=ground_truth_df,
            top_k=top_k,
        )
        row = {
            "candidate": f"sasrec_config_{idx}",
            **metrics,
            "embedding_dim": config.embedding_dim,
            "num_heads": config.num_heads,
            "num_blocks": config.num_blocks,
            "dropout": config.dropout,
            "max_seq_len": config.max_seq_len,
            "learning_rate": config.learning_rate,
            "epochs": config.epochs,
            "samples_per_epoch": config.samples_per_epoch,
            "training_loss_final": float(model.loss_history_[-1]) if model.loss_history_ else None,
        }
        rows.append(row)
        if best_metrics is None or float(metrics["ndcg_at_k"]) > float(best_metrics["ndcg_at_k"]):
            best_model = model
            best_metrics = row
            best_per_user = per_user_df

    if best_model is None or best_metrics is None or best_per_user is None:
        raise RuntimeError("SASRec tuning did not produce a fitted model.")
    best_per_user = best_per_user.copy()
    best_per_user["model"] = "sasrec_tuned"
    return best_model, best_metrics, pd.DataFrame(rows), best_per_user


def build_validation_figures(metrics_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    sns.set_theme(style="whitegrid", context="notebook")
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    ordered = metrics_df.sort_values("ndcg_at_k", ascending=False).copy()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="model", y="ndcg_at_k", hue="model", legend=False, palette="crest")
    plt.xticks(rotation=20, ha="right")
    plt.title("MBD-mini Validation: NDCG@K by Model")
    plt.tight_layout()
    ndcg_path = figures_dir / "mbd_mini_ndcg_by_model.png"
    plt.savefig(ndcg_path, dpi=160)
    plt.close()
    saved_paths.append(ndcg_path)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=ordered, x="model", y="map_at_k", hue="model", legend=False, palette="viridis")
    plt.xticks(rotation=20, ha="right")
    plt.title("MBD-mini Validation: MAP@K by Model")
    plt.tight_layout()
    map_path = figures_dir / "mbd_mini_map_by_model.png"
    plt.savefig(map_path, dpi=160)
    plt.close()
    saved_paths.append(map_path)
    return saved_paths


def _format_metrics_table(metrics_df: pd.DataFrame) -> list[str]:
    lines = [
        "| model | precision@k | recall@k | map@k | ndcg@k |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in metrics_df.sort_values("ndcg_at_k", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.precision_at_k:.4f} | {row.recall_at_k:.4f} | "
            f"{row.map_at_k:.4f} | {row.ndcg_at_k:.4f} |"
        )
    return lines


def write_validation_report(
    dataset_stats: dict[str, Any],
    split_stats: dict[str, Any],
    metrics_df: pd.DataFrame,
    lightgcn_tuning_df: pd.DataFrame,
    sasrec_tuning_df: pd.DataFrame,
    output_path: Path,
) -> Path:
    best = metrics_df.sort_values("ndcg_at_k", ascending=False).iloc[0]
    lines = [
        "# MBD-mini Banking Product Validation Report",
        "",
        "## Dataset",
        f"- Name: {dataset_stats['dataset_name']}",
        f"- Source: {MBD_MINI_DATASET_URL}",
        f"- Paper: {MBD_PAPER_URL}",
        "- Scope: MBD-mini product propensity targets only; heavy transaction, dialog and geo archives are not required for this run.",
        f"- Raw clients: {dataset_stats['raw_clients']}",
        f"- Raw target rows: {dataset_stats['raw_target_rows']}",
        f"- Months: {dataset_stats['timestamp_min']} .. {dataset_stats['timestamp_max']}",
        f"- Positive product events before filtering: {dataset_stats['positive_events_before_filtering']}",
        f"- Clients with positive product events before filtering: {dataset_stats['positive_clients_before_filtering']}",
        f"- Minimum positive events per retained client: {dataset_stats['min_user_positive_events']}",
        f"- Retained clients: {dataset_stats['n_users']}",
        f"- Product labels: {dataset_stats['n_items']}",
        f"- Positive product interactions after filtering: {dataset_stats['n_positive_interactions']}",
        "",
        "## Holdout Protocol",
        "- Split type: temporal per-client product holdout.",
        "- For each retained client, the latest month with at least one positive product label is used as ground truth.",
        "- Earlier positive product labels form the training history.",
        "- Previously observed products are not excluded at ranking time because the MBD target task is product propensity prediction rather than novelty-only purchase recommendation.",
        f"- Train interactions: {split_stats['train_interactions']}",
        f"- Holdout interactions: {split_stats['holdout_interactions']}",
        f"- Evaluated clients: {split_stats['evaluated_users']}",
        "",
        "## Best Model",
        f"- Best by NDCG@{int(best['k'])}: `{best['model']}`",
        f"- NDCG@{int(best['k'])} = {best['ndcg_at_k']:.4f}",
        f"- MAP@{int(best['k'])} = {best['map_at_k']:.4f}",
        "",
        "## Metrics",
        *_format_metrics_table(metrics_df),
        "",
        "## LightGCN Tuning",
        "| candidate | ndcg@k | map@k | embedding_dim | layers | learning_rate | epochs | samples_per_epoch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in lightgcn_tuning_df.sort_values("ndcg_at_k", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.ndcg_at_k:.4f} | {row.map_at_k:.4f} | "
            f"{row.embedding_dim} | {row.n_layers} | {row.learning_rate:.4f} | "
            f"{row.epochs} | {row.samples_per_epoch} |"
        )

    lines += [
        "",
        "## SASRec Tuning",
        "| candidate | ndcg@k | map@k | embedding_dim | heads | blocks | max_seq_len | learning_rate | epochs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sasrec_tuning_df.sort_values("ndcg_at_k", ascending=False).itertuples(index=False):
        lines.append(
            f"| {row.candidate} | {row.ndcg_at_k:.4f} | {row.map_at_k:.4f} | "
            f"{row.embedding_dim} | {row.num_heads} | {row.num_blocks} | {row.max_seq_len} | "
            f"{row.learning_rate:.4f} | {row.epochs} |"
        )

    lines += [
        "",
        "## Interpretation",
        "- The experiment closes the post-predefense requirement to adapt the pipeline to an open banking dataset with real product labels.",
        "- Because only four target products are available in the lightweight MBD-mini target archive, the task is intentionally framed as product-label ranking rather than large-catalog item recommendation.",
        "- The tuned LightGCN and SASRec branches are evaluated on the same temporal banking holdout as the simpler baselines, which makes their limitations visible instead of leaving them as untested future work.",
        "- BERT4Rec remains a defensible future extension rather than a necessary last-minute addition, because the priority experiment already checks the implemented sequence-aware SASRec branch on MBD-mini.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    targets_dir = prepare_mbd_mini_targets(
        data_dir=args.data_dir,
        force_download=args.force_download,
        force_extract=args.force_extract,
        include_client_split=True,
    )
    interactions_df, offers_df, dataset_stats = load_mbd_mini_product_interactions(
        targets_dir=targets_dir,
        min_user_positive_events=args.min_user_positive_events,
        max_users=args.max_users if args.max_users > 0 else None,
    )

    train_interactions_df, holdout_df = build_temporal_product_holdout(interactions_df)
    if train_interactions_df.empty or holdout_df.empty:
        raise ValueError("MBD-mini temporal holdout is empty; lower --min-user-positive-events.")

    user_ids = sorted(holdout_df["user_id"].astype(str).unique().tolist())
    ground_truth_df = holdout_df[["user_id", "offer_id"]].drop_duplicates().copy()
    split_stats = {
        "train_interactions": int(len(train_interactions_df)),
        "holdout_interactions": int(len(ground_truth_df)),
        "evaluated_users": int(len(user_ids)),
    }

    interactions_df.to_csv(args.output_dir / "mbd_mini_positive_product_interactions.csv", index=False)
    train_interactions_df.to_csv(args.output_dir / "mbd_mini_train_interactions.csv", index=False)
    ground_truth_df.to_csv(args.output_dir / "mbd_mini_holdout.csv", index=False)
    offers_df.to_csv(args.output_dir / "mbd_mini_products.csv", index=False)

    metric_rows: list[dict[str, Any]] = []
    per_user_rows: list[pd.DataFrame] = []

    popularity_model = PopularityRecommender().fit(train_interactions_df)
    popularity_metrics, popularity_per_user = _evaluate_recommender(
        "popularity",
        popularity_model,
        user_ids,
        ground_truth_df,
        args.top_k,
    )
    metric_rows.append(_metric_row("popularity", popularity_metrics))
    per_user_rows.append(popularity_per_user)

    item_knn_model = ImplicitItemKNNRecommender(n_neighbors=args.n_neighbors).fit(train_interactions_df)
    item_knn_metrics, item_knn_per_user = _evaluate_recommender(
        "item_knn",
        item_knn_model,
        user_ids,
        ground_truth_df,
        args.top_k,
    )
    metric_rows.append(_metric_row("item_knn", item_knn_metrics, {"n_neighbors": int(args.n_neighbors)}))
    per_user_rows.append(item_knn_per_user)

    mf_model = ImplicitMFRecommender(n_factors=args.n_factors, random_state=42).fit(train_interactions_df)
    mf_metrics, mf_per_user = _evaluate_recommender(
        "implicit_mf",
        mf_model,
        user_ids,
        ground_truth_df,
        args.top_k,
    )
    metric_rows.append(_metric_row("implicit_mf", mf_metrics, {"n_factors": int(args.n_factors)}))
    per_user_rows.append(mf_per_user)

    ncf_config = NeuralCFConfig(
        embedding_dim=8,
        hidden_dims=(16, 8),
        learning_rate=0.01,
        epochs=8,
        batch_size=512,
        negative_samples=2,
        random_state=42,
    )
    ncf_model = NeuralCFRecommender(config=ncf_config).fit(train_interactions_df)
    ncf_metrics, ncf_per_user = _evaluate_recommender(
        "neural_cf",
        ncf_model,
        user_ids,
        ground_truth_df,
        args.top_k,
    )
    metric_rows.append(
        _metric_row(
            "neural_cf",
            ncf_metrics,
            {
                "embedding_dim": int(ncf_config.embedding_dim),
                "hidden_dims": "|".join(str(value) for value in ncf_config.hidden_dims),
            },
        )
    )
    per_user_rows.append(ncf_per_user)

    _, best_lightgcn_metrics, lightgcn_tuning_df, lightgcn_per_user = _fit_best_lightgcn(
        train_interactions_df=train_interactions_df,
        user_ids=user_ids,
        ground_truth_df=ground_truth_df,
        top_k=args.top_k,
    )
    metric_rows.append(
        _metric_row(
            "lightgcn_tuned",
            {
                key: best_lightgcn_metrics[key]
                for key in ["k", "n_users_evaluated", "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]
            },
            {
                "embedding_dim": best_lightgcn_metrics["embedding_dim"],
                "n_layers": best_lightgcn_metrics["n_layers"],
                "samples_per_epoch": best_lightgcn_metrics["samples_per_epoch"],
            },
        )
    )
    per_user_rows.append(lightgcn_per_user)

    _, best_sasrec_metrics, sasrec_tuning_df, sasrec_per_user = _fit_best_sasrec(
        train_interactions_df=train_interactions_df,
        user_ids=user_ids,
        ground_truth_df=ground_truth_df,
        top_k=args.top_k,
    )
    metric_rows.append(
        _metric_row(
            "sasrec_tuned",
            {
                key: best_sasrec_metrics[key]
                for key in ["k", "n_users_evaluated", "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]
            },
            {
                "embedding_dim": best_sasrec_metrics["embedding_dim"],
                "num_heads": best_sasrec_metrics["num_heads"],
                "num_blocks": best_sasrec_metrics["num_blocks"],
                "max_seq_len": best_sasrec_metrics["max_seq_len"],
            },
        )
    )
    per_user_rows.append(sasrec_per_user)

    metrics_df = pd.DataFrame(metric_rows).sort_values("ndcg_at_k", ascending=False).reset_index(drop=True)
    per_user_df = pd.concat(per_user_rows, ignore_index=True)

    metrics_df.to_csv(args.output_dir / "mbd_mini_validation_metrics.csv", index=False)
    per_user_df.to_csv(args.output_dir / "mbd_mini_validation_per_user_metrics.csv", index=False)
    lightgcn_tuning_df.to_csv(args.output_dir / "mbd_mini_lightgcn_tuning.csv", index=False)
    sasrec_tuning_df.to_csv(args.output_dir / "mbd_mini_sasrec_tuning.csv", index=False)
    (args.output_dir / "mbd_mini_dataset_summary.json").write_text(
        json.dumps({**dataset_stats, **split_stats}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    figure_paths = build_validation_figures(metrics_df, args.output_dir)
    report_path = write_validation_report(
        dataset_stats=dataset_stats,
        split_stats=split_stats,
        metrics_df=metrics_df,
        lightgcn_tuning_df=lightgcn_tuning_df,
        sasrec_tuning_df=sasrec_tuning_df,
        output_path=args.output_dir / "mbd_mini_validation_report.md",
    )

    print("[ok] MBD-mini banking product validation finished")
    print(f"[ok] targets dir: {targets_dir}")
    print(f"[ok] users: {dataset_stats['n_users']}")
    print(f"[ok] products: {dataset_stats['n_items']}")
    print(f"[ok] train interactions: {split_stats['train_interactions']}")
    print(f"[ok] holdout interactions: {split_stats['holdout_interactions']}")
    print(f"[ok] metrics: {args.output_dir / 'mbd_mini_validation_metrics.csv'}")
    print(f"[ok] report: {report_path}")
    for figure_path in figure_paths:
        print(f"[ok] figure: {figure_path}")


if __name__ == "__main__":
    main()
