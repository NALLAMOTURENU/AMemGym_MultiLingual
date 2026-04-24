"""
Generate paper-style diagnosis table (Write↓, Read↓, Util.↓ failure rates).
Matches Table 3 / Figure F.3 from the AMemGym paper.

Run after: overall eval + diagnosis for all agents.
"""

import json
import os
import numpy as np


DATA_PATH = "data/v1.base/data.json"

# Each entry: (display_name, agent_config_name, output_dir)
AGENTS = [
    ("LLM (Native)",      "qwen-14b-native",  "eval-output/v1.base/native"),
    ("RAG",               "qwen-14b-rag",      "eval-output/v1.base/rag"),
    ("AWE",               "awe-qwen-local",    "eval-output/v1.base/awe"),
    ("AWI",               "qwen-14b-awi",      "eval-output/v1.base/awi"),
]

with open(DATA_PATH) as f:
    env_data = json.load(f)

# Build {item_id -> num_info_slots} (total required_info entries across all questions)
item_info = {
    item["id"]: {
        "num_info": sum(len(qa["required_info"]) for qa in item["qas"]),
        "num_periods": len(item["periods"]),
    }
    for item in env_data
}


def load_failure_rates(agent_dir_name, output_base):
    """
    Returns (write_failure_rate, read_failure_rate, util_failure_rate) for an agent.

    - write_failure: info never stored correctly (wrong at write time AND read time)
    - read_failure:  info stored but not retrieved (right at write time, wrong at read time)
    - util_failure:  agent correctly recalled state in diagnosis probe but still picked
                     the wrong answer in the overall eval  → measures answer utilization gap
    """
    agent_dir = os.path.join(output_base, agent_dir_name)
    if not os.path.isdir(agent_dir):
        return None, None, None

    write_rates, read_rates, util_rates = [], [], []

    for item in env_data:
        uid = item["id"]
        item_dir = os.path.join(agent_dir, uid)
        diag_metrics_path = os.path.join(item_dir, "diagnosis_metrics.json")
        overall_results_path = os.path.join(item_dir, "overall_results.json")
        diag_results_path = os.path.join(item_dir, "diagnosis_results.json")

        if not os.path.exists(diag_metrics_path):
            print(f"  [missing] {diag_metrics_path}")
            return None, None, None

        dm = json.load(open(diag_metrics_path))
        wf = np.array(dm["write_failure"])   # [Np, Nq]
        rf = np.array(dm["read_failure"])    # [Np, Nq]

        num_info    = item_info[uid]["num_info"]
        num_periods = item_info[uid]["num_periods"]
        total_slots = num_info * num_periods

        write_rates.append(wf.sum() / total_slots)
        read_rates.append(rf.sum()  / total_slots)

        # Util failure: memory_success cases where the overall answer was still wrong
        if os.path.exists(overall_results_path) and os.path.exists(diag_results_path):
            overall = json.load(open(overall_results_path))  # [Np, Nq]
            diag    = json.load(open(diag_results_path))     # [Np, Nq]

            util_fail = 0
            util_total = 0
            for pi in range(num_periods):
                for qi, qa in enumerate(item["qas"]):
                    # Check if diagnosis probe says memory recalled correctly
                    diag_entry = diag[pi][qi]
                    if diag_entry is None:
                        continue
                    # Count info types correctly recalled
                    correct_recalls = sum(
                        1 for r in diag_entry["results"] if r["score"] >= 0.5
                    )
                    n_info = len(diag_entry["results"])
                    mem_ok = (correct_recalls == n_info)  # all info types recalled

                    if mem_ok:
                        # Check if the overall answer was wrong despite correct memory
                        overall_entry = overall[pi][qi]
                        if overall_entry is not None:
                            util_total += 1
                            if overall_entry["scores"]["accuracy"] < 0.5:
                                util_fail += 1
            util_rates.append(util_fail / util_total if util_total > 0 else 0.0)
        else:
            util_rates.append(float("nan"))

    return np.mean(write_rates), np.mean(read_rates), np.nanmean(util_rates)


# ── Collect results ───────────────────────────────────────────────────────────
results = []
for display_name, agent_dir_name, output_base in AGENTS:
    w, r, u = load_failure_rates(agent_dir_name, output_base)
    results.append((display_name, w, r, u))

# ── Print table ───────────────────────────────────────────────────────────────
print()
print("┌─────────────────────┬───────────┬───────────┬───────────┐")
print("│ Strategy            │  Write ↓  │   Read ↓  │   Util. ↓ │")
print("├─────────────────────┼───────────┼───────────┼───────────┤")
for name, w, r, u in results:
    w_s = f"{w:.3f}" if w is not None else "pending"
    r_s = f"{r:.3f}" if r is not None else "pending"
    u_s = f"{u:.3f}" if (u is not None and not np.isnan(u)) else "pending"
    print(f"│ {name:<19} │   {w_s:<6}  │   {r_s:<6}  │   {u_s:<6}  │")
print("└─────────────────────┴───────────┴───────────┴───────────┘")
print()
print("Note: ↓ = lower is better (failure rates)")
print("  Write ↓  — info never correctly stored in memory")
print("  Read  ↓  — info stored but not retrieved/recalled")
print("  Util. ↓  — memory correct but still gave wrong answer")
