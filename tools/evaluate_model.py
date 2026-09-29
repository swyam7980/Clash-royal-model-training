"""
Generates a presentation-ready report (HTML page + PNG charts + CSV table)
showing how well a trained checkpoint imitates your own recorded human
actions.

IMPORTANT: this measures IMITATION ACCURACY, not win rate. For each
recorded (state, action) pair you played, it checks whether the model's
current policy - run in eval mode (argmax, no exploration sampling) on data
it was NOT trained on - would have picked the same origin tile / shell tile
/ hand slot / card identity you actually picked. It does not play any live
games and does not touch BlueStacks.

Requirements (one-time):
    pip install matplotlib --break-system-packages

Usage:
    python -m tools.evaluate_model --model deck1
    python -m tools.evaluate_model --model deck1 --data_dir Resources/Data/HumanGames --out Reports/deck1_eval

Output:
    Reports/<model>/report.html   <- open this in a browser, presentation-ready
    Reports/<model>/*.png         <- individual chart images (also embedded in report.html)
    Reports/<model>/summary.csv   <- raw numbers if you want them in a slide/table elsewhere
"""
import argparse
import csv
import glob
import os
import pickle
from collections import Counter
from datetime import datetime

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

from CRAgent import Agent
from CRHandler import Handler

DATA_DIR_DEFAULT = "Resources/Data/HumanGames"


# ---------------------------------------------------------------- data ----

def load_samples(data_dir):
    samples = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*.pkl"))):
        with open(f, "rb") as fh:
            for s in pickle.load(fh):
                s = dict(s)
                s["_source_file"] = os.path.basename(f)
                samples.append(s)
    return samples


def split_samples(samples, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    n_test = int(len(samples) * test_frac) if len(samples) > 1 else 0
    test_idx = set(idx[:n_test].tolist())
    train, test = [], []
    for i, s in enumerate(samples):
        (test if i in test_idx else train).append(s)
    return train, test


def resolve_card_name(state, slot_idx, distinct_card_names):
    """
    Given a sample's state and a hand-slot index (0-3), returns the card
    name that was actually sitting in that slot at that moment, using the
    one-hot vectors already captured in state['card_data']. Falls back to
    None if it can't be resolved (e.g. deck/model changed since recording).
    """
    try:
        for slot_num, one_hot in state["card_data"]:
            if slot_num - 1 == slot_idx:
                class_idx = int(np.argmax(one_hot)) - 1
                if class_idx < 0:
                    return "(no card)"
                if 0 <= class_idx < len(distinct_card_names):
                    return distinct_card_names[class_idx]
                return None
    except Exception:
        return None
    return None


# ------------------------------------------------------------ evaluate ----

def top_k_accuracy(probs_list, targets, k):
    if not targets:
        return float("nan")
    correct = 0
    for probs, target in zip(probs_list, targets):
        k_eff = min(k, len(probs))
        if target in np.argsort(probs)[-k_eff:]:
            correct += 1
    return correct / len(targets)


def evaluate(agent, samples, distinct_card_names):
    origin_probs_all, shell_probs_all, slot_probs_all = [], [], []
    origin_targets, shell_targets, slot_targets = [], [], []
    card_name_targets, card_name_preds = [], []

    for s in samples:
        enc = agent.state_autoencoder.encode(s["state"])
        origin_probs = agent.origin_actor(enc).numpy()[0]
        shell_probs = agent.shell_actor(enc).numpy()[0]
        slot_probs = agent.card_actor(enc).numpy()[0]

        origin_probs_all.append(origin_probs)
        shell_probs_all.append(shell_probs)
        slot_probs_all.append(slot_probs)

        origin_targets.append(int(s["origin_idx"]))
        shell_targets.append(int(s["shell_idx"]))
        slot_targets.append(int(s["card_idx"]))

        pred_slot = int(np.argmax(slot_probs))
        actual_name = resolve_card_name(s["state"], int(s["card_idx"]), distinct_card_names)
        pred_name = resolve_card_name(s["state"], pred_slot, distinct_card_names)
        if actual_name is not None and pred_name is not None:
            card_name_targets.append(actual_name)
            card_name_preds.append(pred_name)

    def head_stats(probs_all, targets, name):
        preds = [int(np.argmax(p)) for p in probs_all]
        return {
            "name": name,
            "n": len(targets),
            "top1_acc": top_k_accuracy(probs_all, targets, 1),
            "top3_acc": top_k_accuracy(probs_all, targets, 3),
            "preds": preds,
            "targets": targets,
        }

    origin_stats = head_stats(origin_probs_all, origin_targets, "Origin tile (48 tiles + no-action)")
    shell_stats = head_stats(shell_probs_all, shell_targets, "Shell tile (1 of 9)")
    slot_stats = head_stats(slot_probs_all, slot_targets, "Hand slot (1 of 4)")

    exact_all = sum(
        1 for i in range(len(samples))
        if origin_stats["preds"][i] == origin_stats["targets"][i]
        and shell_stats["preds"][i] == shell_stats["targets"][i]
        and slot_stats["preds"][i] == slot_stats["targets"][i]
    )
    overall_action_acc = exact_all / len(samples) if samples else float("nan")

    card_identity_acc = (
        sum(1 for a, b in zip(card_name_targets, card_name_preds) if a == b) / len(card_name_targets)
        if card_name_targets else float("nan")
    )

    return origin_stats, shell_stats, slot_stats, overall_action_acc, card_name_targets, card_name_preds, card_identity_acc


# ---------------------------------------------------------------- plots ----

def plot_head_accuracy(stats_list, card_identity_acc, out_path):
    names = [s["name"] for s in stats_list] + ["Card identity\n(derived)"]
    top1 = [s["top1_acc"] * 100 for s in stats_list] + [card_identity_acc * 100]
    top3 = [s["top3_acc"] * 100 for s in stats_list] + [float("nan")]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width / 2, top1, width, label="Top-1 accuracy", color="#2b5797")
    ax.bar(x + width / 2, top3, width, label="Top-3 accuracy", color="#8fb2e0")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Model accuracy vs. your recorded actions (held-out test data)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    for i, v in enumerate(top1):
        if not np.isnan(v):
            ax.text(i - width / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    for i, v in enumerate(top3):
        if not np.isnan(v):
            ax.text(i + width / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_card_identity_confusion(targets, preds, out_path):
    if not targets:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Not enough data to resolve card identities", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    labels = sorted(set(targets) | set(preds))
    idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    for t, p in zip(targets, preds):
        matrix[idx[t]][idx[p]] += 1

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Card model would have played")
    ax.set_ylabel("Card you actually played")
    ax.set_title("Card-identity confusion matrix (derived)")
    thresh = matrix.max() / 2 if matrix.max() else 0
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                         color="white" if matrix[i, j] > thresh else "black", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_dataset_composition(card_name_targets, out_path):
    if not card_name_targets:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Not enough data to resolve card identities", ha="center", va="center")
        ax.axis("off")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    counts = Counter(card_name_targets)
    labels = sorted(counts, key=lambda l: -counts[l])
    values = [counts[l] for l in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel("Recorded samples (test split)")
    ax.set_title("Recorded data - which cards you actually played")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i, v in enumerate(values):
        ax.text(i, v + 0.05, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sessions_over_time(samples, out_path):
    by_file = Counter(s.get("_source_file", "unknown") for s in samples)
    files = sorted(by_file.keys())
    counts = [by_file[f] for f in files]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(files)), counts, color="#55A868")
    ax.set_xticks(range(len(files)))
    ax.set_xticklabels(files, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Actions recorded")
    ax.set_title("Recorded actions by session file (all data, train+test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------- html ----

def build_html_report(out_dir, model_name, data_dir, n_train, n_test,
                       origin_stats, shell_stats, slot_stats,
                       overall_action_acc, card_identity_acc,
                       chart_files, generated_at):
    rows = ""
    for s in (origin_stats, shell_stats, slot_stats):
        rows += f"""
        <tr>
            <td>{s['name']}</td>
            <td>{s['n']}</td>
            <td>{s['top1_acc']*100:.1f}%</td>
            <td>{s['top3_acc']*100:.1f}%</td>
        </tr>"""
    card_id_row = f"""
        <tr>
            <td>Card identity (derived)</td>
            <td>{n_test}</td>
            <td>{card_identity_acc*100:.1f}%</td>
            <td>-</td>
        </tr>""" if not np.isnan(card_identity_acc) else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{model_name} - Model Evaluation Report</title>
<style>
    body {{ font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; color: #222; background: #fafafa; }}
    h1 {{ margin-bottom: 4px; }}
    .meta {{ color: #666; margin-bottom: 32px; font-size: 14px; }}
    .summary-cards {{ display: flex; gap: 16px; margin-bottom: 32px; flex-wrap: wrap; }}
    .card {{ background: white; border-radius: 10px; padding: 18px 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 170px; }}
    .card .value {{ font-size: 28px; font-weight: 700; color: #2b5797; }}
    .card .label {{ font-size: 13px; color: #666; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 12px;}}
    th, td {{ padding: 10px 16px; text-align: left; border-bottom: 1px solid #eee; }}
    th {{ background: #2b5797; color: white; font-weight: 600; }}
    .chart {{ background: white; padding: 16px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 28px; }}
    .chart img {{ width: 100%; max-width: 720px; display: block; margin: 0 auto; }}
    .note {{ font-size: 13px; color: #888; margin: 8px 0 28px; }}
</style>
</head>
<body>
    <h1>Model Evaluation: {model_name}</h1>
    <div class="meta">Generated {generated_at} &middot; data source: {data_dir}</div>

    <div class="summary-cards">
        <div class="card"><div class="value">{n_train}</div><div class="label">Training samples</div></div>
        <div class="card"><div class="value">{n_test}</div><div class="label">Held-out test samples</div></div>
        <div class="card"><div class="value">{overall_action_acc*100:.1f}%</div><div class="label">Full-action match rate (all 3 heads correct)</div></div>
        <div class="card"><div class="value">{card_identity_acc*100:.1f}%</div><div class="label">Card identity accuracy</div></div>
    </div>

    <table>
        <tr><th>Model head</th><th>Test samples</th><th>Top-1 accuracy</th><th>Top-3 accuracy</th></tr>
        {rows}{card_id_row}
    </table>
    <div class="note">"Accuracy" = does the model's current policy (evaluated deterministically, no exploration)
    pick the same tile/slot/card you actually picked, on data it was <b>not</b> trained on. This measures
    imitation of your recorded play, not win rate against an opponent. "Card identity" is derived by
    cross-referencing the predicted/actual hand slot against that sample's own on-screen cards, since the
    model's card head predicts a hand-slot position (1-4), not a fixed card identity.</div>

    <div class="chart"><img src="{chart_files['head_accuracy']}"></div>
    <div class="chart"><img src="{chart_files['card_confusion']}"></div>
    <div class="chart"><img src="{chart_files['dataset_composition']}"></div>
    <div class="chart"><img src="{chart_files['sessions']}"></div>
</body>
</html>
"""
    with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


def write_summary_csv(out_dir, origin_stats, shell_stats, slot_stats,
                       overall_action_acc, card_identity_acc, n_train, n_test):
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["training_samples", n_train])
        writer.writerow(["test_samples", n_test])
        writer.writerow(["overall_full_action_accuracy", f"{overall_action_acc:.4f}"])
        writer.writerow(["card_identity_accuracy", f"{card_identity_acc:.4f}"])
        for s in (origin_stats, shell_stats, slot_stats):
            writer.writerow([f"{s['name']}_top1_accuracy", f"{s['top1_acc']:.4f}"])
            writer.writerow([f"{s['name']}_top3_accuracy", f"{s['top3_acc']:.4f}"])


# ----------------------------------------------------------------- main ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="base", help="checkpoint folder name under TrainedWeights/ (default: base)")
    parser.add_argument("--data_dir", default=DATA_DIR_DEFAULT, help="folder of recorded .pkl sessions to evaluate against")
    parser.add_argument("--out", default=None, help="output folder (default: Reports/<model>/)")
    parser.add_argument("--test_frac", type=float, default=0.2, help="fraction of samples held out for testing")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_path = "TrainedWeights/" if args.model in (None, "base") else f"TrainedWeights/{args.model}/"
    out_dir = args.out or os.path.join("Reports", args.model)
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading checkpoint: {model_path}")
    agent = Agent(load=False)
    agent.load(path=model_path)

    print(f"Loading recorded samples from: {args.data_dir}")
    samples = load_samples(args.data_dir)
    if not samples:
        print("ERROR: no recorded sessions found - run tools/recorder.py first.")
        return
    print(f"Loaded {len(samples)} recorded actions.")

    train_samples, test_samples = split_samples(samples, test_frac=args.test_frac, seed=args.seed)
    if not test_samples:
        print("WARNING: not enough samples for a held-out test split - evaluating on all data instead. "
              "Accuracy will look better than true generalization would.")
        test_samples = samples
        train_samples = samples

    env = Handler(load_elixir_model=False)
    distinct_card_names = env.get_distinct_card_names()

    print(f"Evaluating on {len(test_samples)} held-out samples...")
    (origin_stats, shell_stats, slot_stats, overall_action_acc,
     card_name_targets, card_name_preds, card_identity_acc) = evaluate(agent, test_samples, distinct_card_names)

    print("\n=== RESULTS ===")
    for s in (origin_stats, shell_stats, slot_stats):
        print(f"{s['name']:<36} top-1: {s['top1_acc']*100:5.1f}%   top-3: {s['top3_acc']*100:5.1f}%   (n={s['n']})")
    print(f"{'Card identity (derived)':<36} top-1: {card_identity_acc*100:5.1f}%")
    print(f"{'Full action match (all 3 heads)':<36} {overall_action_acc*100:5.1f}%")

    print("\nGenerating charts...")
    chart_files = {
        "head_accuracy": "head_accuracy.png",
        "card_confusion": "card_identity_confusion.png",
        "dataset_composition": "dataset_composition.png",
        "sessions": "sessions_over_time.png",
    }
    plot_head_accuracy([origin_stats, shell_stats, slot_stats], card_identity_acc,
                        os.path.join(out_dir, chart_files["head_accuracy"]))
    plot_card_identity_confusion(card_name_targets, card_name_preds,
                                  os.path.join(out_dir, chart_files["card_confusion"]))
    plot_dataset_composition(card_name_targets, os.path.join(out_dir, chart_files["dataset_composition"]))
    plot_sessions_over_time(samples, os.path.join(out_dir, chart_files["sessions"]))

    build_html_report(
        out_dir, args.model, args.data_dir, len(train_samples), len(test_samples),
        origin_stats, shell_stats, slot_stats, overall_action_acc, card_identity_acc,
        chart_files, datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    write_summary_csv(out_dir, origin_stats, shell_stats, slot_stats,
                       overall_action_acc, card_identity_acc, len(train_samples), len(test_samples))

    print(f"\nReport saved to: {os.path.join(out_dir, 'report.html')}")
    print("Open it in a browser - it's presentation-ready with charts, tables and summary cards.")
    os._exit(0)


if __name__ == "__main__":
    main()
