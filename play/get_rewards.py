# https://manage.unitynodes.io/rewards/allocation
# Network tab in browser dev tools (check Preserve logs and disable cache)
# look for "rewards_get_allocations" POST request
# get the new Bearer token from the request headers if expired

import requests
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
URL = "https://vtllpagtmncbkywsqccd.supabase.co/rest/v1/rpc/rewards_get_allocations"
SECRETS_PATH = r"D:\dev\unity\unity-rewards\play\secrets.json"

with open(SECRETS_PATH, "r") as f:
    secrets = json.load(f)

API_KEY = secrets["apikey"]
BEARER_TOKEN = secrets["bearer"]

HEADERS = {
    "accept": "*/*",
    "apikey": API_KEY,
    "authorization": "Bearer {BEARER_TOKEN}",
    "content-profile": "public",
    "content-type": "application/json",
    "origin": "https://manage.unitynodes.io",
}

PAYLOAD = {
    "skip": None,
    "take": None
}

# -------------------------------------------------
# OUTPUT DIRECTORY (customise this)
# -------------------------------------------------
TARGET_DIR = r"D:\dev\unity\unity-rewards\play\output"

# -------------------------------------------------
# FUNCTIONS
# -------------------------------------------------

def fetch_allocations():
    r = requests.post(URL, headers=HEADERS, json=PAYLOAD)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame(data)
    df = df[["licenseId", "licenseLeaseId", "completedAt", "amountMicros"]]
    df["completedAt"] = pd.to_datetime(df["completedAt"])
    return df


def plot_amount_distribution(df, out_path=None, in_dollars=True):
    if out_path is None:
        out_path = TARGET_DIR + r"\amount_distribution.png"

    s = df["amountMicros"].astype(float)
    if in_dollars:
        s = s / 1e6
        xlabel = "Reward amount (units, 1e6 = 1.0)"
    else:
        xlabel = "amountMicros"

    plt.figure(figsize=(8, 5))
    plt.hist(s, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Number of rewards")
    plt.title("Distribution of rewards")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_amount_over_time(df, out_path=None, in_dollars=True):
    if out_path is None:
        out_path = TARGET_DIR + r"\amount_over_time.png"

    s = df["amountMicros"].astype(float)
    if in_dollars:
        s = s / 1e6
        ylabel = "Reward amount (units, 1e6 = 1.0)"
    else:
        ylabel = "amountMicros"

    plt.figure(figsize=(10, 5))
    plt.scatter(df["completedAt"], s, s=10, alpha=0.6)
    plt.xlabel("Completed at")
    plt.ylabel(ylabel)
    plt.title("Rewards over time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def load_boxes():
    boxes_path = TARGET_DIR + r"\boxes.csv"
    df_boxes = pd.read_csv(boxes_path)
    return df_boxes


def merge_rewards_boxes(df_rewards, df_boxes):
    return df_rewards.merge(df_boxes, on="licenseId", how="left")


def compute_box_stats(df_merged, in_dollars=True):
    df_tmp = df_merged.copy()
    s = df_tmp["amountMicros"].astype(float)
    if in_dollars:
        s = s / 1e6
    df_tmp["amountUnits"] = s

    grp = df_tmp.groupby("boxId")["amountUnits"]
    stats_df = pd.DataFrame(
        {
            "total_reward": grp.sum(),
            "count_rewards": grp.count(),
            "avg_reward": grp.mean(),
        }
    ).reset_index()

    return stats_df


def plot_box_stats_table(stats_df, out_path=None):
    if out_path is None:
        out_path = TARGET_DIR + r"\box_stats_table.png"

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(stats_df))))
    ax.axis("off")

    table = ax.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(stats_df.columns))))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_box_pie(stats_df, out_path=None):
    if out_path is None:
        out_path = TARGET_DIR + r"\box_total_rewards_pie.png"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(stats_df["total_reward"], labels=stats_df["boxId"], autopct="%1.1f%%")
    ax.set_title("Total rewards per boxId")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_license_distribution(df_merged, out_path=None, in_dollars=True):
    if out_path is None:
        out_path = TARGET_DIR + r"\license_distribution_by_box.png"

    df_plot = df_merged.copy()

    s = df_plot["amountMicros"].astype(float)
    if in_dollars:
        s = s / 1e6
    df_plot["amountUnits"] = s

    df_plot["license_short"] = df_plot["licenseId"].astype(str).str[-4:]

    box_ids = sorted(df_plot["boxId"].dropna().unique())
    n_plots = len(box_ids) + 1

    fig, axes = plt.subplots(n_plots, 1, figsize=(14, 3.2 * n_plots), sharex=False)
    if n_plots == 1:
        axes = [axes]

    # --- Each Box ---
    for idx, box_id in enumerate(box_ids):
        ax = axes[idx]
        sub = df_plot[df_plot["boxId"] == box_id]

        agg_amount = (
            sub.groupby("license_short")["amountUnits"]
            .sum()
            .sort_values(ascending=False)
        )

        agg_count = (
            sub.groupby("license_short")["amountUnits"]
            .count()
            .reindex(agg_amount.index)
        )

        x = range(len(agg_amount))

        ax2 = ax.twinx()

        ax.bar(x, agg_amount.values, color="steelblue", label="Total Reward")
        ax2.bar([i + 0.25 for i in x], agg_count.values,
                color="orange", width=0.25, label="Count")

        ax.set_title(f"Box {box_id}")
        ax.set_ylabel("Total Reward")
        ax2.set_ylabel("Reward Count")

        ax.set_xticks([i + 0.125 for i in x])
        ax.set_xticklabels(agg_amount.index, rotation=90, fontsize=6)

    # --- All Boxes Combined ---
    ax_all = axes[-1]
    agg_amount_all = (
        df_plot.groupby("license_short")["amountUnits"]
        .sum()
        .sort_values(ascending=False)
    )
    agg_count_all = (
        df_plot.groupby("license_short")["amountUnits"]
        .count()
        .reindex(agg_amount_all.index)
    )

    x = range(len(agg_amount_all))
    ax2_all = ax_all.twinx()

    ax_all.bar(x, agg_amount_all.values, color="steelblue", label="Total Reward")
    ax2_all.bar([i + 0.25 for i in x], agg_count_all.values,
                color="orange", width=0.25, label="Count")

    ax_all.set_title("All Boxes Combined")
    ax_all.set_ylabel("Total Reward")
    ax2_all.set_ylabel("Reward Count")

    ax_all.set_xticks([i + 0.125 for i in x])
    ax_all.set_xticklabels(agg_amount_all.index, rotation=90, fontsize=6)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    df_rewards = fetch_allocations()
    df_rewards_csv = TARGET_DIR + r"\rewards.csv"
    df_rewards.to_csv(df_rewards_csv, index=False)

    df_boxes = load_boxes()
    df_merged = merge_rewards_boxes(df_rewards, df_boxes)
    df_merged_csv = TARGET_DIR + r"\rewards_with_boxes.csv"
    df_merged.to_csv(df_merged_csv, index=False)

    plot_amount_distribution(df_rewards)
    plot_amount_over_time(df_rewards)

    stats_df = compute_box_stats(df_merged)
    plot_box_stats_table(stats_df)
    plot_box_pie(stats_df)

    plot_license_distribution(df_merged)

    print(df_merged.head())
    print(f"Saved: {df_rewards_csv}")
    print(f"Saved: {df_merged_csv}")
    print(f"Saved PNGs in {TARGET_DIR}")


if __name__ == "__main__":
    main()
