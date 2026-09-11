# -*- coding: utf-8 -*-
"""
Fig. 5 — Decision reliability by city

Creates a publication-quality scatter/transition plot for AARS 1-shot vs 3-shot
using the strict city-level external metrics.

Expected CSV columns:
    target_city
    anchor_budget
    target
    design_regret_kwh_m2
    top_recovery

Usage:
    python make_fig05_decision_reliability_by_city.py \
        --input external_metrics_by_city_strict.csv \
        --output-dir figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CITY_DISPLAY = {
    "Ardahan": "Ardahan",
    "Bartin": "Bartın",
    "Bingol": "Bingöl",
    "Canakkale": "Çanakkale",
    "Hatay": "Hatay",
    "Kars": "Kars",
    "Konya": "Konya",
    "Sanliurfa": "Şanlıurfa",
}


def _spread_labels(
    y_values: np.ndarray,
    ymin: float,
    ymax: float,
    min_gap: float,
) -> np.ndarray:
    """
    Redistribute label y-positions so nearby labels do not overlap.
    Keeps ordering and tries to remain close to original data points.
    """
    order = np.argsort(y_values)
    sorted_y = y_values[order].astype(float).copy()

    # Forward pass
    for i in range(1, len(sorted_y)):
        sorted_y[i] = max(sorted_y[i], sorted_y[i - 1] + min_gap)

    # Pull back if top boundary is exceeded
    if sorted_y[-1] > ymax:
        sorted_y -= sorted_y[-1] - ymax

    # Backward pass to preserve minimum spacing
    for i in range(len(sorted_y) - 2, -1, -1):
        sorted_y[i] = min(sorted_y[i], sorted_y[i + 1] - min_gap)

    # Pull up if bottom boundary is exceeded
    if sorted_y[0] < ymin:
        sorted_y += ymin - sorted_y[0]

    result = np.empty_like(sorted_y)
    result[order] = sorted_y
    return result


def make_figure(input_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(input_csv)

    required = {
        "target_city",
        "anchor_budget",
        "target",
        "design_regret_kwh_m2",
        "top_recovery",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Missing required CSV columns: " + ", ".join(sorted(missing))
        )

    # Only total-energy AARS strict results
    data = df[df["target"].astype(str).str.lower().eq("total")].copy()
    data = data[data["anchor_budget"].isin([1, 3])].copy()

    if data.empty:
        raise ValueError("No total-energy 1-shot / 3-shot rows found.")

    one = (
        data[data["anchor_budget"].eq(1)]
        .set_index("target_city")
        [["design_regret_kwh_m2", "top_recovery"]]
        .rename(
            columns={
                "design_regret_kwh_m2": "regret_1",
                "top_recovery": "recovery_1",
            }
        )
    )

    three = (
        data[data["anchor_budget"].eq(3)]
        .set_index("target_city")
        [["design_regret_kwh_m2", "top_recovery"]]
        .rename(
            columns={
                "design_regret_kwh_m2": "regret_3",
                "top_recovery": "recovery_3",
            }
        )
    )

    plot_df = one.join(three, how="inner").reset_index()

    if len(plot_df) != 8:
        raise ValueError(
            f"Expected 8 held-out climates, found {len(plot_df)}."
        )

    # Convert recovery fraction to percentage.
    plot_df["recovery_1_pct"] = 100.0 * plot_df["recovery_1"]
    plot_df["recovery_3_pct"] = 100.0 * plot_df["recovery_3"]

    # Reproducible, journal-oriented typography.
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )

    fig, ax = plt.subplots(figsize=(9.0, 6.2))

    # Draw city-wise transition from 1-shot to 3-shot.
    for _, row in plot_df.iterrows():
        ax.plot(
            [row["regret_1"], row["regret_3"]],
            [row["recovery_1_pct"], row["recovery_3_pct"]],
            linewidth=0.9,
            alpha=0.40,
            zorder=1,
        )

    ax.scatter(
        plot_df["regret_1"],
        plot_df["recovery_1_pct"],
        marker="o",
        s=55,
        zorder=3,
        label="AARS 1-shot",
    )

    ax.scatter(
        plot_df["regret_3"],
        plot_df["recovery_3_pct"],
        marker="s",
        s=58,
        zorder=4,
        label="AARS 3-shot",
    )

    # Mean operating points.
    mean_1 = (
        plot_df["regret_1"].mean(),
        plot_df["recovery_1_pct"].mean(),
    )
    mean_3 = (
        plot_df["regret_3"].mean(),
        plot_df["recovery_3_pct"].mean(),
    )

    ax.scatter(*mean_1, marker="X", s=85, zorder=5)
    ax.scatter(*mean_3, marker="P", s=85, zorder=5)

    ax.annotate(
        "mean 1-shot",
        mean_1,
        xytext=(7, -12),
        textcoords="offset points",
        fontsize=8.0,
    )
    ax.annotate(
        "mean 3-shot",
        mean_3,
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=8.0,
    )

    # Establish axis limits before calculating label positions.
    all_x = np.r_[
        plot_df["regret_1"].to_numpy(),
        plot_df["regret_3"].to_numpy(),
    ]
    all_y = np.r_[
        plot_df["recovery_1_pct"].to_numpy(),
        plot_df["recovery_3_pct"].to_numpy(),
    ]

    x_span = max(all_x.max() - all_x.min(), 1.0)
    y_span = max(all_y.max() - all_y.min(), 1.0)

    x_left = max(0.0, all_x.min() - 0.05 * x_span)
    x_data_right = all_x.max() + 0.04 * x_span

    # Leave a dedicated label column on the right.
    x_label = x_data_right + 0.14 * x_span
    x_right = x_label + 0.24 * x_span

    y_bottom = all_y.min() - 0.06 * y_span
    y_top = all_y.max() + 0.08 * y_span

    ax.set_xlim(x_left, x_right)
    ax.set_ylim(y_bottom, y_top)

    # ------------------------------------------------------------------
    # Collision-free city labels
    # ------------------------------------------------------------------
    # Only 3-shot points are labeled. Labels are placed in a clean right
    # column and connected back to their exact data point with leader lines.
    # This prevents Kars / Şanlıurfa / Hatay and other nearby labels from
    # overlapping without moving any data points.
    label_y = _spread_labels(
        plot_df["recovery_3_pct"].to_numpy(),
        ymin=y_bottom + 0.03 * y_span,
        ymax=y_top - 0.03 * y_span,
        min_gap=max(0.90, 0.038 * (y_top - y_bottom)),
    )

    for (_, row), y_text in zip(plot_df.iterrows(), label_y):
        x_point = row["regret_3"]
        y_point = row["recovery_3_pct"]
        city = CITY_DISPLAY.get(row["target_city"], row["target_city"])

        ax.plot(
            [x_point, x_label - 0.015 * x_span],
            [y_point, y_text],
            linewidth=0.65,
            alpha=0.55,
            zorder=2,
        )

        ax.text(
            x_label,
            y_text,
            city,
            ha="left",
            va="center",
            fontsize=8.2,
            zorder=6,
        )

    ax.set_xlabel("Design regret (kWh/m²-year)  ← lower is better")
    ax.set_ylabel("Top-5% design recovery (%)  ↑ higher is better")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.12)

    ax.text(
        0.02,
        0.98,
        "Preferred region: higher recovery and lower regret",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
    )

    # Legend with marker semantics.
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            markersize=7,
            label="AARS 1-shot",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markersize=7,
            label="AARS 3-shot",
        ),
        Line2D(
            [0], [0],
            linestyle="-",
            linewidth=0.9,
            label="Same climate",
        ),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "fig05_decision_reliability_by_city.pdf"
    png_path = output_dir / "fig05_decision_reliability_by_city.png"

    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=600)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Fig. 5 decision-reliability plot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/results/strict/external_metrics_by_city_strict.csv",
        help="Path to external_metrics_by_city_strict.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "figures/generated",
        help="Output directory (default: figures)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.input, args.output_dir)
