import os
import json
import argparse
import collections
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logging
from evaluation import eval_design
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def is_valid_seq(seq, structure):
    """
    Check if the sequence is valid given the target structure.
    The sequence is invalid if:
      - Its length does not match the structure.
      - The structural pairs are not valid (must be one of: CG, GC, AU, UA, GU, UG).
      - The structure contains unbalanced or invalid characters.
    """
    if len(seq) != len(structure):
        return False

    valid_pairs = {"CG", "GC", "AU", "UA", "GU", "UG"}
    stack = []

    for idx, symbol in enumerate(structure):
        if symbol == '(':
            stack.append(idx)
        elif symbol == ')':
            if not stack:
                return False
            pair_idx = stack.pop()
            pair = seq[pair_idx] + seq[idx]
            if pair not in valid_pairs:
                return False
        elif symbol != '.':
            return False

    if stack:
        return False

    return True

def load_records(path):
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if 'best_design_per_turn' not in rec:
                rec['best_design_per_turn'] = [rec['designed_sequence']]
            records.append(rec)
    #random.shuffle(records)
    #records = records[:2500]
    return records

def group_by_id(records):
    d = collections.defaultdict(list)
    for rec in records:
        d[rec['id']].append(rec)
    return d

def evaluate_pair(args):
    """
    Wrapper for eval_design to unpack arguments.
    args: tuple (sequence, target_structure)
    """
    seq, ss = args
    return eval_design(seq, ss)

def evaluate_all(records, desc, n_workers, cache_df, cache_index, batch_size=10000):
    """
    Evaluate all records with caching, in chunks of `batch_size` pairs.
    This avoids submitting millions of tasks at once.

    Parameters:
      - records: list of dicts, each with 'id', 'target_structure',
                 'designed_sequence', and 'best_design_per_turn'.
      - desc: description string for logging/tqdm.
      - n_workers: number of parallel workers.
      - cache_df: Pandas DataFrame of cached eval_design results.
      - cache_index: set of (target_structure, sequence) tuples already cached.
      - batch_size: how many (seq, tgt) pairs to process in one chunk.

    Returns:
      - valid_records: list of records (with 'metrics' and 'turn_metrics' added)
      - cache_df: updated DataFrame with newly computed rows appended
      - cache_index: updated set with new (target_structure, sequence) tuples
    """
    num_input = len(records)
    logger.info(f"{desc}: Starting evaluation of {num_input} total records.")

    # 1) Separate valid vs invalid; add "all-A" fallback
    invalid_records = [
        rec for rec in records
        if not is_valid_seq(rec["designed_sequence"], rec["target_structure"])
    ]
    valid_records = [
        rec for rec in records
        if is_valid_seq(rec["designed_sequence"], rec["target_structure"])
    ]
    logger.info(f"{desc}: Found {len(valid_records)} valid records, {len(invalid_records)} invalid records.")

    valid_ids = {rec["id"] for rec in valid_records}
    all_ids = {rec["id"] for rec in records}
    missing_ids = all_ids - valid_ids

    if missing_ids:
        logger.info(f"{desc}: Adding all-'A' fallback for {len(missing_ids)} missing IDs.")
        for mid in missing_ids:
            placeholder = next(rec for rec in invalid_records if rec["id"] == mid)
            length = len(placeholder["target_structure"])
            baseline_seq = "A" * length
            baseline_rec = {
                **placeholder,
                "designed_sequence": baseline_seq,
                "_was_allA_fallback": True,
            }
            valid_records.append(baseline_rec)

    # 2) Gather all (target_structure, sequence) pairs
    all_pairs = set()
    for rec in valid_records:
        tgt = rec["target_structure"]
        seq_main = rec["designed_sequence"]
        all_pairs.add((tgt, seq_main))
        for seq_turn in rec["best_design_per_turn"]:
            all_pairs.add((tgt, seq_turn))
    logger.info(f"{desc}: Collected {len(all_pairs)} unique (structure, sequence) pairs.")

    # 3) Identify which pairs are missing from cache
    missing_pairs = [(seq, tgt) for (tgt, seq) in all_pairs if (tgt, seq) not in cache_index]
    logger.info(f"{desc}: {len(missing_pairs)} pairs missing from cache.")

    # 4) If there are missing pairs, process them in batches
    if missing_pairs:
        logger.info(f"{desc}: Starting parallel evaluation in batches of {batch_size}, with {n_workers} workers each.")
        new_rows = []

        total_batches = (len(missing_pairs) + batch_size - 1) // batch_size
        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(missing_pairs))
            batch = missing_pairs[start:end]
            logger.info(f"{desc}: Processing batch {batch_idx+1}/{total_batches} (pairs {start} to {end-1}).")

            # Use executor.map on this batch
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                for metrics in tqdm(
                    executor.map(evaluate_pair, batch),
                    total=len(batch),
                    desc=f"{desc} (batch {batch_idx+1}/{total_batches})",
                    unit="pair"
                ):
                    tgt = metrics["target_structure"]
                    seq = metrics["sequence"]
                    row = {
                        "target_structure": tgt,
                        "sequence": seq,
                        "mfe_structures": metrics["mfe_structures"],
                        "structural_dist": metrics["structural_dist"],
                        "probability": metrics["probability"],
                        "ensemble_defect": metrics["ensemble_defect"],
                        "energy_diff": metrics["energy_diff"],
                        "is_mfe": metrics["is_mfe"],
                        "is_umfe": metrics["is_umfe"],
                    }
                    new_rows.append(row)
                    cache_index.add((tgt, seq))

        if new_rows:
            logger.info(f"{desc}: Appending {len(new_rows)} new rows to cache.")
            new_df = pd.DataFrame(new_rows)
            if cache_df.empty:
                cache_df = new_df.copy()
            else:
                cache_df = pd.concat([cache_df, new_df], ignore_index=True)
        else:
            logger.info(f"{desc}: No new rows generated; cache unchanged.")
    else:
        logger.info(f"{desc}: No missing pairs; skipping evaluation.")

    # 5) Build lookup dict from cache_df
    logger.info(f"{desc}: Building cache lookup dictionary ({len(cache_df)} total cached rows).")
    cache_lookup = {
        (row.target_structure, row.sequence): {
            "mfe_structures": row.mfe_structures,
            "structural_dist": row.structural_dist,
            "probability": row.probability,
            "ensemble_defect": row.ensemble_defect,
            "energy_diff": row.energy_diff,
            "is_mfe": bool(row.is_mfe),
            "is_umfe": bool(row.is_umfe),
            "sequence": row.sequence,
            "target_structure": row.target_structure
        }
        for row in cache_df.itertuples()
    }

    # 6) Assign metrics + turn_metrics to each valid_record
    logger.info(f"{desc}: Assigning metrics and turn_metrics to {len(valid_records)} valid records.")
    for rec in tqdm(valid_records, desc=f"{desc} (assigning)", unit="rec"):
        tgt = rec["target_structure"]
        seq_main = rec["designed_sequence"]
        rec["metrics"] = cache_lookup[(tgt, seq_main)]
        rec["turn_metrics"] = [
            cache_lookup[(tgt, seq_turn)]
            for seq_turn in rec["best_design_per_turn"]
        ]

    logger.info(f"{desc}: Finished assigning metrics.")
    return valid_records, cache_df, cache_index

def compute_sota_metrics(sota_groups):
    solve_rates, dists, probs, neds = [], [], [], []
    for runs in tqdm(sota_groups.values(), desc="Computing SOTA metrics"):
        ml = [r['metrics'] for r in runs]
        solve_rates.append(max(m['is_mfe'] for m in ml))
        dists.append(min(m['structural_dist'] for m in ml))
        neds.append(min(m['ensemble_defect'] for m in ml))
        probs.append(max(m['probability'] for m in ml))
    return {
        'solve_rate': np.mean(solve_rates),
        'structural_dist': np.mean(dists),
        'ensemble_defect': np.mean(neds),
        'probability': np.mean(probs),
    }

def best_of_n_curve(groups, max_n, metric, log_base=None, num_points=100):
    """
    Compute a (possibly subsampled) best-of-N curve.

    If log_base is set (e.g. 2 or 10), we'll only evaluate the curve
    at `num_points` N‐values spaced evenly on a log scale between 1 and N_max.
    Otherwise we do every integer N from 1 to N_max.
    """
    # figure out the maximum n we can actually do
    min_runs = min(len(runs) for runs in groups.values())
    max_plot = min(max_n, min_runs)

    # pick which N's to evaluate
    if log_base:
        # exponent range: log_base^0 = 1 up to log_base^E = max_plot
        max_exp = np.log(max_plot) / np.log(log_base)
        raw = np.logspace(0, max_exp, num=num_points, base=log_base)
        n_list = np.unique(np.round(raw).astype(int))
        n_list = n_list[(n_list >= 1) & (n_list <= max_plot)]
    else:
        n_list = np.arange(1, max_plot+1, dtype=int)

    xs, ys = [], []
    for n in tqdm(n_list, desc="Best-of-N points", leave=False):
        vals = []
        for runs in groups.values():
            subset = runs[:n]
            if metric == 'solve_rate':
                v = max(r['metrics']['is_mfe'] for r in subset)
            elif metric == 'structural_dist':
                v = min(r['metrics']['structural_dist'] for r in subset)
            elif metric == 'ensemble_defect':
                v = min(r['metrics']['ensemble_defect'] for r in subset)
            else:  # 'probability'
                v = max(r['metrics']['probability'] for r in subset)
            vals.append(v)
        xs.append(int(n))
        ys.append(np.mean(vals))

    return xs, ys

def _beautify(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    #ax.legend(frameon=False, fontsize=10)
    #plt.tight_layout()
    # place legend outside on the right
    ax.legend(frameon=False,
            fontsize=10,
            bbox_to_anchor=(1.02, 1),
            loc='upper left')

    # leave room on the right for the legend
    plt.tight_layout(rect=[0, 0, 1, 1])

def plot_best_of_n(groups_list, exp_names, sota_vals, max_n, out_dir, log_base=None):
    #colors = ["#0072B2", "#DDD025", "#CA408C", "#009E73", "#D55E00", "#56B4E9"]
    #colors = ["#0072B2", "#DDD025", "#CA408C", "#451267", "#892F64", "#CF5D41", "#F0AA4F", "#01010A"]
    colors = ["#0072B2", "#451267", "#CF5D41", "#F0AA4F", "#F0C84F", "#01010A", "#892F64"]
    for metric, title in [
        ('solve_rate',      'Solve Rate'),
        ('structural_dist', 'Structural Distance'),
        ('ensemble_defect', 'Normalized Ensemble Defect'),
        ('probability',     'Probability'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, (groups, name) in enumerate(zip(groups_list, exp_names)):
            xs, ys = best_of_n_curve(
                groups,
                max_n,
                metric,
                log_base=log_base,
                num_points=50   # e.g. 100 points on the log curve
            )
            ax.plot(xs, ys,
                    marker='x', markersize=4,
                    color=colors[i],
                    linewidth=1.5, alpha=0.8,
                    label=name)
            # draw model-specific best performance line
            if metric in ['structural_dist', 'ensemble_defect']:
                best_val = min(ys)
            else:
                best_val = max(ys)
            ax.axhline(best_val,
                       linestyle='--', linewidth=1.5,
                       color=colors[i], alpha=0.8,
                       label=f"{name} (Best)")

        # SOTA baseline
        ax.axhline(sota_vals[metric],
                   linestyle='--', linewidth=1.5,
                   color='red', label='SAMFEO')
        if metric in ['structural_dist', 'ensemble_defect']:
            ax.invert_yaxis()

        if log_base:
            ax.set_xscale('log', base=log_base)

        _beautify(ax,
                  xlabel='Number of Samples (N)',
                  ylabel=title,
                  title=f'Best-of-N {title}')
        fig.savefig(os.path.join(out_dir, f'best_of_N_{metric}.png'))
        fig.savefig(os.path.join(out_dir, f'best_of_N_{metric}.pdf'))
        plt.close(fig)


def plot_turns_histogram(groups_list, exp_names, out_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    for groups, name in zip(groups_list, exp_names):
        lengths = [len(r['best_design_per_turn'])
                   for runs in groups.values()
                   for r in runs]
        ax.hist(lengths, bins=20,
                alpha=0.5, edgecolor='black',
                label=name)
    _beautify(ax,
              xlabel='Number of Turns',
              ylabel='Frequency',
              title='Distribution of Number of Turns')
    fig.savefig(os.path.join(out_dir, 'turns_histogram.png'))
    plt.close(fig)

def performance_vs_turns(groups_list, exp_names, sota_vals, Ns, max_turns, out_dir):
    linestyles = ['-', ':', '--', '-.', (0, (1, 1)), (0, (5, 1))]
    cmap = plt.get_cmap('viridis')

    for metric, title in [
        ('solve_rate',      'Solve Rate'),
        ('structural_dist', 'Structural Distance'),
        ('ensemble_defect', 'Normalized Ensemble Defect'),
        ('probability',     'Probability'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 6))

        for exp_idx, (groups, name) in enumerate(zip(groups_list, exp_names)):
            ls = linestyles[exp_idx % len(linestyles)]
            colors = cmap(np.linspace(0.1, 0.9, len(Ns)))

            # figure out how many runs each ID actually has
            #min_runs = min(len(runs) for runs in groups.values())
            max_runs = max(len(runs) for runs in groups.values())

            for i, N in enumerate(Ns):
                if N > max_runs:
                    # skip plotting N if this experiment never ran that many times
                    continue

                xs, ys = [], []
                for k in range(1, max_turns+1):
                    vals = []
                    for runs in groups.values():
                        subset = runs[:N]
                        per_turn = [
                            r['turn_metrics'][k-1] if k-1 < len(r['turn_metrics']) else r['turn_metrics'][-1]
                            for r in subset
                        ]
                        if metric == 'solve_rate':
                            v = max(m['is_mfe'] for m in per_turn)
                        elif metric == 'structural_dist':
                            v = min(m['structural_dist'] for m in per_turn)
                        elif metric == 'ensemble_defect':
                            v = min(m['ensemble_defect'] for m in per_turn)
                        else:
                            v = max(m['probability'] for m in per_turn)
                        vals.append(v)
                    xs.append(k)
                    ys.append(np.mean(vals))

                ax.plot(xs, ys,
                        linestyle=ls,
                        color=colors[i],
                        marker='x', markersize=2,
                        linewidth=1.5, alpha=0.9,
                        label=f"{name}, N={N}")

        ax.axhline(sota_vals[metric],
                   linestyle='--', linewidth=1.5,
                   color='red', label='SAMFEO')
        if metric in ['structural_dist', 'ensemble_defect']:
            ax.invert_yaxis()

        _beautify(ax,
                  xlabel='Turn',
                  ylabel=title,
                  title=f'{title} vs Turn')

        fig.savefig(os.path.join(out_dir,
                                 f'performance_vs_turns_{metric}.png'))
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple RNA-design experiments")
    parser.add_argument(
        '--results',
        nargs='+',
        default=[],
        help="List of JSONL files for each experiment")
    parser.add_argument(
        '--exp_names',
        nargs='+',
        default=[],
        help="List of (same-length) names for each experiment")
    parser.add_argument(
        '--sota',
        type=str,
        default=[],
        help="Path to SOTA JSONL")
    parser.add_argument(
        '--out_dir',
        default='../plots_bon/',
        help="Directory to save plots")
    parser.add_argument(
        '--max_n',
        type=int,
        default=1000,
        help="Max runs for Best-of-N")
    parser.add_argument(
        '--max_turns',
        type=int,
        default=10,
        help="Max turns to plot")
    parser.add_argument(
        '--n_workers',
        type=int,
        default=50,
        help="Number of processes for evaluation")
    parser.add_argument(
        '--log-base', type=int, choices=[2, 10], default=None,
        help="Plot x-axis in log scale with the given base (2 or 10)")
    parser.add_argument(
        '--cache_path',
        type=str,
        default='./eval_cache_sept.parquet',
        help="Path to cache parquet file (stores previously computed eval_design results)")
    args = parser.parse_args()

    if len(args.results) != len(args.exp_names):
        parser.error("`--results` and `--exp_names` must have the same number of entries")

    os.makedirs(args.out_dir, exist_ok=True)

    # load or initialize cache
    if args.cache_path:
        if os.path.exists(args.cache_path):
            cache_df = pd.read_parquet(args.cache_path)
        else:
            cache_df = pd.DataFrame(columns=[
                "target_structure", "sequence",
                "mfe_structures", "structural_dist", "probability",
                "ensemble_defect", "energy_diff", "is_mfe", "is_umfe"
            ])
        # Use a set of tuples for fast lookup
        cache_index = set(
            zip(cache_df["target_structure"].astype(str),
                cache_df["sequence"].astype(str))
        )
    else:
        cache_df = pd.DataFrame(columns=[
            "target_structure", "sequence",
            "mfe_structures", "structural_dist", "probability",
            "ensemble_defect", "energy_diff", "is_mfe", "is_umfe"
        ])
        cache_index = set()

    # load & evaluate each experiment
    groups_list = []
    for path, exp_name in zip(args.results, args.exp_names):
        records = load_records(path)

        valid_records, cache_df, cache_index = evaluate_all(
            records=records,
            desc=f"Evaluating {os.path.basename(path)}",
            n_workers=args.n_workers,
            cache_df=cache_df,
            cache_index=cache_index
        )

        groups_list.append(group_by_id(valid_records))

    # load & evaluate SOTA
    sota_records = load_records(args.sota)
    valid_sota_records, cache_df, cache_index = evaluate_all(
        records=sota_records,
        desc="Evaluating SOTA",
        n_workers=args.n_workers,
        cache_df=cache_df,
        cache_index=cache_index
    )
    sota_groups = group_by_id(valid_sota_records)
    print("Computing SOTA metrics …")
    sota_vals = compute_sota_metrics(sota_groups)

    # save updated cache
    if args.cache_path:
        os.makedirs(os.path.dirname(args.cache_path), exist_ok=True)
        cache_df.to_parquet(args.cache_path, index=False)

    # plotting
    plot_best_of_n(groups_list, args.exp_names, sota_vals, args.max_n, args.out_dir, log_base=args.log_base)
    exit(0)
    plot_turns_histogram(groups_list, args.exp_names, args.out_dir)
    performance_vs_turns(groups_list, args.exp_names, sota_vals,
                         Ns=[1,10,30,50, 100], max_turns=args.max_turns,
                         out_dir=args.out_dir)

if __name__ == "__main__":
    main()
