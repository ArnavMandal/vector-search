"""Plot measured recall and tail latency from a calibrated compare.py run."""
import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, MultipleLocator


def main():
    benchmarks = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=benchmarks / "high_recall.jsonl")
    parser.add_argument("--output", type=Path, default=benchmarks / "high_recall.svg")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines()]
    shapes = sorted({(r["rows"], r["dim"]) for r in rows})
    targets = {r["recall_target"] for r in rows}
    if len(targets) != 1:
        parser.error("the input must contain one calibration target")
    target = targets.pop() * 100

    styles = {"native-hnsw": ("Native HNSW", "#08696c", "o"),
              "faiss-hnsw": ("FAISS HNSW", "#ae4d23", "s")}
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "svg.hashsalt": "vector-search"})
    fig, axes = plt.subplots(1, len(shapes), figsize=(5.1 * len(shapes), 3.9), sharey=True, squeeze=False)
    for ax, (n, dim) in zip(axes[0], shapes):
        group = [r for r in rows if (r["rows"], r["dim"]) == (n, dim)]
        ax.axhline(target, color="#727b82", linestyle="--", linewidth=1, zorder=1)
        for engine, (_, color, marker) in styles.items():
            sweep = sorted((r for r in group if r["engine"] == engine), key=lambda r: r["efSearch"])
            ax.plot([r["p95_us"] / 1000 for r in sweep], [r["recall_at_10"] * 100 for r in sweep],
                    color=color, marker=marker, markersize=4, linewidth=1.6)
            for r in sweep:
                if r["selected"]:
                    ax.plot(r["p95_us"] / 1000, r["recall_at_10"] * 100, "o", markersize=11,
                            markerfacecolor="none", markeredgecolor=color, markeredgewidth=1.4)
        for r in group:
            if r["engine"] == "faiss-flat":
                ax.plot(r["p95_us"] / 1000, r["recall_at_10"] * 100, "x", color="#272e35", markersize=7)
        selected = {r["efSearch"] for r in group if r["selected"]}
        title = f"{n:,} vectors × {dim} dimensions"
        if len(selected) == 1:
            title += f"\nSelected efSearch = {selected.pop()}"
        ax.set_title(title, fontsize=11)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
        ax.tick_params(which="minor", bottom=False)
        ax.grid(axis="y", alpha=0.19)
        ax.spines[["top", "right"]].set_visible(False)

    low = max(0, 5 * math.floor(min(r["recall_at_10"] * 100 for r in rows) / 5))
    axes[0, 0].set_ylim(min(low, target - 5) - 1, 102)
    axes[0, 0].yaxis.set_major_locator(MultipleLocator(10 if low < 65 else 5))
    axes[0, 0].set_ylabel("Held-out Recall@10 (%)")
    fig.supxlabel("Single-query p95 latency (ms, log scale)", y=0.17, fontsize=10)
    legend = [Line2D([], [], color=c, marker=m, label=name) for name, c, m in styles.values()]
    legend.extend([Line2D([], [], color="#272e35", marker="x", linestyle="none", label="FAISS exact"),
                   Line2D([], [], color="#555", marker="o", fillstyle="none", linestyle="none",
                          markersize=9, label="Selected on calibration"),
                   Line2D([], [], color="#727b82", linestyle="--", label=f"{target:g}% recall target")])
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.075, bottom=0.32, right=0.985, top=0.91, wspace=0.14)
    metadata = {"Date": None} if args.output.suffix.lower() == ".svg" else None
    fig.savefig(args.output, metadata=metadata, dpi=155, bbox_inches="tight", pad_inches=0.13)
    plt.close(fig)


if __name__ == "__main__":
    main()
