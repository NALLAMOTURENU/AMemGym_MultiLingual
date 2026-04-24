"""
Generate paper-style figures from existing eval-output results.
Produces heatmaps matching Figure 3 style from the AMemGym paper.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ── Config ──────────────────────────────────────────────────────────────────
DATA_PATH      = "data/v1.base/data.json"
AWI_OUTPUT_DIR = "eval-output/v1.base/awi/qwen-14b-awi"
UB_METRICS     = "eval-output/v1.base/upperbound/qwen-local/utilization_metrics.json"
FIGURES_DIR    = "figures/results"
AGENT_LABEL    = "AWI (Qwen-2.5-32B)"

os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Load env data ────────────────────────────────────────────────────────────
with open(DATA_PATH) as f:
    env_data = json.load(f)

item_ids = [item["id"] for item in env_data]
max_periods = max(len(item["periods"]) for item in env_data)
nq = len(env_data[0]["qas"])
N  = len(env_data)


# ── Compute random baseline ──────────────────────────────────────────────────
def compute_random_scores(data):
    per_user = []
    for item in data:
        periods_scores = []
        for period in item["periods"]:
            cur_state = period["state"]
            qa_scores = []
            for qa in item["qas"]:
                required = [cur_state[k] for k in qa["required_info"]]
                hit = [1.0 if c["state"] == required else 0.0 for c in qa["answer_choices"]]
                qa_scores.append(np.mean(hit))
            periods_scores.append(qa_scores)
        per_user.append(periods_scores)
    return per_user  # ragged [Nu][Np_i][Nq]

random_per_user = compute_random_scores(env_data)


# ── Load AWI overall_metrics ──────────────────────────────────────────────────
awi_per_user = []
for item_id in item_ids:
    with open(os.path.join(AWI_OUTPUT_DIR, item_id, "overall_metrics.json")) as f:
        awi_per_user.append(json.load(f)["accuracy"])  # [Np_i, Nq]


# ── Load upperbound utilization_metrics ──────────────────────────────────────
with open(UB_METRICS) as f:
    ub_raw = json.load(f)["accuracy"]  # [N, max_Np, Nq] with NaN padding

ub_arr = np.array(ub_raw)  # shape (N, max_Np, Nq)


# ── Per-period aggregation (handle variable period lengths) ──────────────────
awi_per_period  = []
rand_per_period = []
ub_per_period   = []

for p in range(max_periods):
    awi_v, rand_v, ub_v = [], [], []
    for u in range(N):
        if p < len(awi_per_user[u]):
            awi_v.extend(awi_per_user[u][p])
            rand_v.extend(random_per_user[u][p])
            if not np.isnan(ub_arr[u, p, 0]):
                ub_v.extend(ub_arr[u, p, :].tolist())
    awi_per_period.append(np.mean(awi_v)   if awi_v  else float("nan"))
    rand_per_period.append(np.mean(rand_v) if rand_v else float("nan"))
    ub_per_period.append(np.mean(ub_v)     if ub_v   else float("nan"))

awi_mean  = np.nanmean(awi_per_period)
rand_mean = np.nanmean(rand_per_period)
ub_mean   = np.nanmean(ub_per_period)

# Normalized memory score: (acc - random) / (UB - random)
mem_per_period = [
    (a - r) / (u - r) if (not np.isnan(a) and (u - r) > 1e-9) else float("nan")
    for a, r, u in zip(awi_per_period, rand_per_period, ub_per_period)
]
mem_mean = np.nanmean(mem_per_period)


# ── Helpers ───────────────────────────────────────────────────────────────────
def fmt(x):
    if np.isnan(x): return "N/A"
    return f"{x:.3f}".lstrip('0') if x < 1 else f"{x:.2f}"

def make_heatmap(rows_data, row_labels, x_labels, title, cbar_label,
                 output_path, vmin=0.0, vmax=1.0, cmap=plt.cm.RdYlGn):
    plt.clf()
    n_rows = len(row_labels)
    fig, ax = plt.subplots(figsize=(max(7.5, len(x_labels) * 0.52), 1.2 + 0.55 * n_rows))
    arr   = np.array(rows_data, dtype=float)
    annot = np.vectorize(fmt)(arr)
    mask  = np.isnan(arr)
    sns.heatmap(arr, ax=ax, annot=annot, fmt='', mask=mask,
                vmin=vmin, vmax=vmax, cmap=cmap,
                yticklabels=row_labels, xticklabels=x_labels,
                annot_kws={"size": 9}, linewidths=0.3)
    ax.collections[0].colorbar.set_label(label=cbar_label, size=11, weight='bold')
    ax.set_xlabel("Period Index", fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='x', labelsize=8)
    plt.title(title, fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ── Figure 1: Overall Accuracy Heatmap (paper Fig 3 left) ────────────────────
# Columns: [UB, Mean, 0, 1, ..., 20]  — matches the paper exactly
x_labels_overall = ["UB", "Mean"] + [str(i) for i in range(max_periods)]

ub_row   = [float("nan"), ub_mean]   + ub_per_period
awi_row  = [float("nan"), awi_mean]  + awi_per_period
rand_row = [float("nan"), rand_mean] + rand_per_period

make_heatmap(
    rows_data  = [ub_row, awi_row, rand_row],
    row_labels = ["UB", AGENT_LABEL, "Random"],
    x_labels   = x_labels_overall,
    title      = "Overall Accuracy Score",
    cbar_label = "Overall Score",
    output_path = os.path.join(FIGURES_DIR, "overall_accuracy.png"),
    vmin=0.0, vmax=1.0,
)


# ── Figure 2: Normalized Memory Score Heatmap (paper Fig 3 right) ────────────
# Columns: [Mean, 0, 1, ..., 20]
x_labels_mem = ["Mean"] + [str(i) for i in range(max_periods)]

mem_awi_row = [mem_mean] + mem_per_period

make_heatmap(
    rows_data  = [mem_awi_row],
    row_labels = [AGENT_LABEL],
    x_labels   = x_labels_mem,
    title      = "Memory Score  (acc − random) / (UB − random)",
    cbar_label = "Memory Score",
    output_path = os.path.join(FIGURES_DIR, "memory_score.png"),
    vmin=0.0, vmax=1.0,
)


# ── Figure 3: Per-user accuracy trajectory ────────────────────────────────────
plt.clf()
fig, ax = plt.subplots(figsize=(11, 4))
colors = plt.cm.Set2(np.linspace(0, 1, N))

for u_idx in range(N):
    periods   = range(len(awi_per_user[u_idx]))
    accs      = [np.mean(awi_per_user[u_idx][p]) for p in periods]
    rand_accs = [np.mean(random_per_user[u_idx][p]) for p in periods]
    ub_accs   = [np.nanmean(ub_arr[u_idx, p, :]) for p in periods]
    ax.plot(periods, accs,      marker='o', markersize=4, linewidth=1.5,
            color=colors[u_idx], label=f"User {u_idx+1} (AWI)")
    ax.plot(periods, rand_accs, linewidth=1, linestyle='--',
            color=colors[u_idx], alpha=0.4)
    ax.plot(periods, ub_accs,   linewidth=1, linestyle=':',
            color=colors[u_idx], alpha=0.6)

ax.set_xlabel("Period Index", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Per-User Accuracy over Time — AWI solid · UB dotted · Random dashed",
             fontsize=11, fontweight='bold')
ax.set_ylim(-0.05, 1.05)
ax.set_xticks(range(max_periods))
ax.legend(fontsize=8, ncol=3, loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(FIGURES_DIR, "per_user_trajectory.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# ── Figure 4: Per-question accuracy heatmap ───────────────────────────────────
per_q_per_period = np.full((max_periods, nq), np.nan)
for p in range(max_periods):
    for q in range(nq):
        vals = [awi_per_user[u][p][q] for u in range(N) if p < len(awi_per_user[u])]
        if vals:
            per_q_per_period[p, q] = np.mean(vals)

plt.clf()
fig, ax = plt.subplots(figsize=(12, 5))
annot = np.vectorize(fmt)(per_q_per_period.T)
sns.heatmap(per_q_per_period.T, ax=ax, annot=annot, fmt='',
            vmin=0.0, vmax=1.0, cmap=plt.cm.RdYlGn,
            yticklabels=[f"Q{i+1}" for i in range(nq)],
            xticklabels=[str(i) for i in range(max_periods)],
            annot_kws={"size": 7}, linewidths=0.3)
ax.collections[0].colorbar.set_label(label='Accuracy', size=11, weight='bold')
ax.set_xlabel("Period Index", fontsize=11)
ax.set_ylabel("Question", fontsize=11)
ax.set_title("Per-Question Accuracy over Time — AWI (Qwen-2.5-32B)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
out = os.path.join(FIGURES_DIR, "per_question_heatmap.png")
plt.savefig(out, dpi=300, bbox_inches='tight')
plt.savefig(out.replace(".png", ".pdf"), dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("RESULTS SUMMARY  (paper Table 2 format)")
print("=" * 65)
print(f"{'Agent':<28} {'Overall':>9}  {'UB':>7}  {'Random':>8}  {'Memory↑':>9}")
print("-" * 65)
print(f"{'AWI (Qwen-2.5-32B)':<28} {awi_mean:>9.4f}  {ub_mean:>7.4f}  {rand_mean:>8.4f}  {mem_mean:>9.4f}")
print(f"{'Upper-Bound (UB)':<28} {ub_mean:>9.4f}  {'—':>7}  {'—':>8}  {'—':>9}")
print(f"{'Random Baseline':<28} {rand_mean:>9.4f}  {'—':>7}  {'—':>8}  {'—':>9}")
print("=" * 65)
print(f"\nMemory Score = (AWI − Random) / (UB − Random)")
print(f"            = ({awi_mean:.4f} − {rand_mean:.4f}) / ({ub_mean:.4f} − {rand_mean:.4f})")
print(f"            = {mem_mean:.4f}")
