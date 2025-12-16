import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

RPC_URL = "https://vtllpagtmncbkywsqccd.supabase.co/rest/v1/rpc/rewards_get_allocations"

st.set_page_config(page_title="Unity Rewards Dashboard", layout="wide")
st.title("Unity Rewards Dashboard")
st.markdown("Paste your Bearer token, upload `boxes.csv`, then click **Fetch & Analyse**.")

# ------------------------------------------------------------
# GLOBAL DATE FILTER
# ------------------------------------------------------------
default_from = date(2025, 11, 30)
default_to = date.today()

c_from, c_to = st.columns(2)
with c_from:
    from_date = st.date_input("From date", value=default_from, format="DD/MM/YYYY")
with c_to:
    to_date = st.date_input("To date", value=default_to, format="DD/MM/YYYY")

col1, col2 = st.columns(2)
with col1:
    bearer = st.text_input("Bearer token", type="password")
with col2:
    apikey = st.text_input(
        "Supabase API key (anon)",
        value="sb_publishable_yKqi0fu5vV6G4ryUIMJuzw_NCoFEl1c",
    )

boxes_file = st.file_uploader(
    "Upload boxes.csv (phoneID, licenseId, boxId, boxType)",
    type=["csv"],
)

run_btn = st.button("Fetch & Analyse")
status_placeholder = st.empty()


# ------------------------------------------------------------
# FETCH REWARDS (PAGINATED)
# ------------------------------------------------------------
def fetch_rewards(bearer_token, api_key, page_size=1000):
    headers = {
        "accept": "*/*",
        "apikey": api_key,
        "authorization": f"Bearer {bearer_token}",
        "content-type": "application/json",
        "content-profile": "public",
    }

    all_rows = []
    skip = 0

    while True:
        payload = {"skip": skip, "take": page_size}
        r = requests.post(RPC_URL, headers=headers, json=payload)
        r.raise_for_status()

        batch = r.json()
        if not batch:
            break

        all_rows.extend(batch)

        if len(batch) < page_size:
            break

        skip += page_size

    if not all_rows:
        return pd.DataFrame(columns=["licenseId", "licenseLeaseId", "completedAt", "amountMicros", "date", "amountUnits"])

    df = pd.DataFrame(all_rows)
    df = df[["licenseId", "licenseLeaseId", "completedAt", "amountMicros"]]
    df["completedAt"] = pd.to_datetime(df["completedAt"])
    df["date"] = df["completedAt"].dt.date
    df["amountMicros"] = df["amountMicros"].astype(float)
    df["amountUnits"] = df["amountMicros"] / 1e6
    return df


# ------------------------------------------------------------
# LOAD BOX DATA
# ------------------------------------------------------------
def load_boxes(file):
    df = pd.read_csv(file)
    expected = {"licenseId", "boxId"}
    if not expected.issubset(df.columns):
        raise ValueError("boxes.csv must contain at least licenseId and boxId columns")
    return df


# ------------------------------------------------------------
# BOX STATISTICS
# ------------------------------------------------------------
def compute_box_stats(df_merged):
    grp = df_merged.groupby("boxId")["amountUnits"]
    stats_df = pd.DataFrame(
        {
            "boxType": df_merged.groupby("boxId")["boxType"].first(),
            "total_reward": grp.sum(),
            "count_rewards": grp.count(),
            "avg_reward": grp.mean(),
        }
    ).reset_index()
    stats_df = stats_df.sort_values("total_reward", ascending=False)
    return stats_df


# ------------------------------------------------------------
# LICENSE DISTRIBUTION BY BOX (kept for dual-axis chart if needed)
# ------------------------------------------------------------
def build_license_stats(df_merged):
    df = df_merged.copy()
    df["license_short"] = df["licenseId"].astype(str).str[-4:]

    by_box = {}
    for box_id, sub in df.groupby("boxId"):
        agg = (
            sub.groupby("license_short")["amountUnits"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "total", "count": "count"})
        )
        by_box[box_id] = agg.sort_values("total", ascending=False)

    all_agg = (
        df.groupby("license_short")["amountUnits"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "total", "count": "count"})
        .sort_values("total", ascending=False)
    )
    by_box["__ALL__"] = all_agg
    return by_box


def make_dual_axis_bar(agg, title):
    labels = agg.index.tolist()
    totals = agg["total"].tolist()
    counts = agg["count"].tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=labels, y=totals, name="Total reward", opacity=0.8), secondary_y=False)
    fig.add_trace(go.Bar(x=labels, y=counts, name="Reward count", opacity=0.7), secondary_y=True)

    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title="License (last 4 chars)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, l=40, r=40, b=80),
    )
    fig.update_yaxes(title_text="Total reward (units)", secondary_y=False)
    fig.update_yaxes(title_text="Count", secondary_y=True)
    return fig


# ------------------------------------------------------------
# MAIN EXECUTION
# ------------------------------------------------------------
if run_btn:
    try:
        if from_date > to_date:
            st.error("From date must be earlier than (or equal to) To date.")
            st.stop()

        if not bearer.strip():
            st.error("Please paste a Bearer token.")
            st.stop()
        if not apikey.strip():
            st.error("Please provide an API key.")
            st.stop()
        if not boxes_file:
            st.error("Please upload boxes.csv.")
            st.stop()

        status_placeholder.info("Loading boxes.csv...")
        df_boxes = load_boxes(boxes_file)

        status_placeholder.info("Fetching rewards from Supabase (paginated)...")
        df_rewards_all = fetch_rewards(bearer, apikey)

        df_rewards = df_rewards_all[
            (df_rewards_all["date"] >= from_date) & (df_rewards_all["date"] <= to_date)
        ].copy()

        status_placeholder.info("Merging rewards with boxes...")
        df_merged = df_rewards.merge(df_boxes, on="licenseId", how="left")

        st.success(f"Fetched {len(df_rewards_all)} rewards total, {len(df_rewards)} in selected date range.")

        # ------------------------------------------------------------
        # SUMMARY METRICS
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Summary")

        total_rewards = df_rewards["amountUnits"].sum()
        total_days = df_rewards["date"].nunique()
        total_jobs = len(df_rewards)
        unique_phones = df_merged["phoneID"].dropna().nunique()
        avg_jobs_per_phone = total_jobs / unique_phones if unique_phones > 0 else 0
        avg_reward = df_rewards["amountUnits"].mean() if total_jobs > 0 else 0

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total rewards", f"{total_rewards:,.6f}")
        c2.metric("Total days", f"{total_days}")
        c3.metric("Total jobs", f"{total_jobs:,}")
        c4.metric("Avg jobs per phone", f"{avg_jobs_per_phone:,.2f}")
        c5.metric("Avg reward amount", f"{avg_reward:,.6f}")

        # ------------------------------------------------------------
        # TOP 10 REWARDS
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Top 10 rewards (within selected period)")

        top10 = (
            df_merged[["date", "phoneID", "boxId", "boxType", "amountUnits"]]
            .sort_values("amountUnits", ascending=False)
            .head(10)
            .rename(columns={"amountUnits": "rewardAmount"})
        )
        st.dataframe(top10)

        st.subheader("Top 10 rewards (excluding cellular)")

        mask_non_cell = df_merged["boxType"].fillna("").str.lower() != "cellular"
        top10_non_cell = (
            df_merged.loc[mask_non_cell, ["date", "phoneID", "boxId", "boxType", "amountUnits"]]
            .sort_values("amountUnits", ascending=False)
            .head(10)
            .rename(columns={"amountUnits": "rewardAmount"})
        )
        st.dataframe(top10_non_cell)

        # ------------------------------------------------------------
        # DOWNLOADS
        # ------------------------------------------------------------
        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download rewards.csv (filtered)",
                df_rewards.to_csv(index=False).encode("utf-8"),
                "rewards.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                "Download rewards_with_boxes.csv (filtered)",
                df_merged.to_csv(index=False).encode("utf-8"),
                "rewards_with_boxes.csv",
                mime="text/csv",
            )

        # ------------------------------------------------------------
        # BOX STATISTICS
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Box statistics")

        stats_df = compute_box_stats(df_merged)
        st.dataframe(stats_df)

        col_pie, col_hist = st.columns(2)

        with col_pie:
            fig_pie = px.pie(
                stats_df,
                names="boxId",
                values="total_reward",
                title="Total rewards per boxId",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_hist:
            fig_hist = px.histogram(
                df_rewards,
                x="amountUnits",
                nbins=50,
                title="Reward amount distribution (units)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # ------------------------------------------------------------
        # DISTRIBUTION BY PHONEID (TOP 5 BOXES)
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Distribution by phone (top 5 boxes)")

        top_boxes = (
            stats_df[["boxId", "boxType", "total_reward"]]
            .dropna(subset=["boxId"])
            .head(5)
            .copy()
        )

        if top_boxes.empty:
            st.info("No box data available in the selected period.")
        else:
            box_ids = top_boxes["boxId"].tolist()

            df_box_phone = df_merged.dropna(subset=["boxId", "phoneID"]).copy()
            df_box_phone = df_box_phone[df_box_phone["boxId"].isin(box_ids)]

            agg_box_phone = (
                df_box_phone.groupby(["boxId", "phoneID"])["amountUnits"]
                .sum()
                .reset_index()
                .rename(columns={"amountUnits": "totalReward"})
            )

            cols = st.columns(2)
            for i, row in enumerate(top_boxes.itertuples(index=False)):
                box_id = row.boxId
                box_type = row.boxType

                plot_df = agg_box_phone[agg_box_phone["boxId"] == box_id].sort_values("totalReward", ascending=False)

                fig = px.bar(
                    plot_df,
                    x="phoneID",
                    y="totalReward",
                    title=f"Box {box_id} ({box_type}) - total rewards by phone",
                    labels={"phoneID": "Phone ID", "totalReward": "Total rewards (units)"},
                )
                fig.update_layout(margin=dict(t=60, l=40, r=40, b=80))

                if i < 4:
                    with cols[i % 2]:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.plotly_chart(fig, use_container_width=True)

        # ------------------------------------------------------------
        # REWARDS OVER TIME (scatter)
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Rewards over time")

        fig_time = px.scatter(
            df_rewards,
            x="completedAt",
            y="amountUnits",
            opacity=0.6,
            title="Rewards over time",
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # ------------------------------------------------------------
        # PHONE EARNINGS PER DAY
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Phone earnings per day (sum of daily rewards)")

        df_phone = df_merged.dropna(subset=["phoneID"]).copy()
        daily_phone = (
            df_phone.groupby(["phoneID", "date"])["amountUnits"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        fig_phone = px.line(
            daily_phone,
            x="date",
            y="amountUnits",
            color="phoneID",
            title="Daily earnings per phone",
        )
        fig_phone.update_layout(
            hovermode="x unified",
            legend_title_text="Phone ID",
            margin=dict(t=60, l=40, r=40, b=40),
        )
        st.plotly_chart(fig_phone, use_container_width=True)

        # ------------------------------------------------------------
        # BOX EARNINGS PER DAY
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Box earnings per day (sum of daily rewards)")

        df_box = df_merged.dropna(subset=["boxId"]).copy()
        daily_box = (
            df_box.groupby(["boxId", "date"])["amountUnits"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        fig_box = px.line(
            daily_box,
            x="date",
            y="amountUnits",
            color="boxId",
            title="Daily earnings per box",
        )
        fig_box.update_layout(
            hovermode="x unified",
            legend_title_text="Box ID",
            margin=dict(t=60, l=40, r=40, b=40),
        )
        st.plotly_chart(fig_box, use_container_width=True)

        status_placeholder.empty()

    except Exception as e:
        st.error(f"Error: {e}")
        status_placeholder.empty()
