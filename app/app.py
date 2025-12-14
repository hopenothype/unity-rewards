import io
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

RPC_URL = "https://vtllpagtmncbkywsqccd.supabase.co/rest/v1/rpc/rewards_get_allocations"

st.set_page_config(page_title="Unity Rewards Dashboard", layout="wide")

st.title("Unity Rewards Dashboard")

st.markdown(
    "Paste your Bearer token, upload `boxes.csv`, then click **Fetch & Analyse**."
)

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
# FETCH REWARDS
# ------------------------------------------------------------
def fetch_rewards(bearer_token, api_key):
    headers = {
        "accept": "*/*",
        "apikey": api_key,
        "authorization": f"Bearer {bearer_token}",
        "content-type": "application/json",
        "content-profile": "public",
    }
    payload = {"skip": None, "take": None}
    r = requests.post(RPC_URL, headers=headers, json=payload)
    r.raise_for_status()

    data = r.json()
    df = pd.DataFrame(data)
    df = df[["licenseId", "licenseLeaseId", "completedAt", "amountMicros"]]
    df["completedAt"] = pd.to_datetime(df["completedAt"])
    df["date"] = df["completedAt"].dt.date    # ← NEW
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
    # Group by box and calculate metrics
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
# LICENSE DISTRIBUTION BY BOX
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

    fig.add_trace(
        go.Bar(x=labels, y=totals, name="Total reward", opacity=0.8),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=labels, y=counts, name="Reward count", opacity=0.7),
        secondary_y=True,
    )

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

        status_placeholder.info("Fetching rewards from Supabase...")
        df_rewards = fetch_rewards(bearer, apikey)

        status_placeholder.info("Merging rewards with boxes...")
        df_merged = df_rewards.merge(df_boxes, on="licenseId", how="left")

        st.success(f"Fetched {len(df_rewards)} rewards.")

        # ------------------------------------------------------------
        # DOWNLOADS
        # ------------------------------------------------------------
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                "Download rewards.csv",
                df_rewards.to_csv(index=False).encode("utf-8"),
                "rewards.csv",
                mime="text/csv",
            )
        with col_b:
            st.download_button(
                "Download rewards_with_boxes.csv",
                df_merged.to_csv(index=False).encode("utf-8"),
                "rewards_with_boxes.csv",
                mime="text/csv",
            )

        st.subheader("Raw merged data (head)")
        st.dataframe(df_merged.head())


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
                stats_df, names="boxId", values="total_reward",
                title="Total rewards per boxId"
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_hist:
            fig_hist = px.histogram(
                df_rewards, x="amountUnits", nbins=50,
                title="Reward amount distribution (units)"
            )
            st.plotly_chart(fig_hist, use_container_width=True)


        # ------------------------------------------------------------
        # LICENSE DISTRIBUTION (fixed dropdown)
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("License distribution by box")

        license_stats = build_license_stats(df_merged)

        box_options = [("__ALL__", "All boxes")] + [
            (b, str(b)) for b in license_stats.keys() if b != "__ALL__"
        ]
        box_dict = {label: value for (value, label) in box_options}

        chosen_label = st.selectbox("Select box", list(box_dict.keys()))
        chosen_box = box_dict[chosen_label]

        agg = license_stats[chosen_box]
        fig_dual = make_dual_axis_bar(
            agg,
            "All boxes" if chosen_box == "__ALL__" else f"Box {chosen_box}"
        )
        st.plotly_chart(fig_dual, use_container_width=True)


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
        # PHONE EARNINGS PER DAY (NEW AGGREGATION)
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Phone earnings per day (sum of daily rewards)")

        df_phone = df_merged.dropna(subset=["phoneID"]).copy()

        # Daily sum per phone
        daily_phone = (
            df_phone.groupby(["phoneID", "date"])["amountUnits"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        # Cumulative
        daily_phone["cum"] = daily_phone.groupby("phoneID")["amountUnits"].cumsum()

        y_col = "amountUnits"

        fig_phone = px.line(
            daily_phone,
            x="date",
            y=y_col,
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
        # BOX EARNINGS PER DAY (NEW AGGREGATION)
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("Box earnings per day (sum of daily rewards)")

        df_box = df_merged.dropna(subset=["boxId"]).copy()

        # Daily sum per box
        daily_box = (
            df_box.groupby(["boxId", "date"])["amountUnits"]
            .sum()
            .reset_index()
            .sort_values("date")
        )

        # Cumulative
        daily_box["cum"] = daily_box.groupby("boxId")["amountUnits"].cumsum()

        y_col2 = "amountUnits"

        fig_box = px.line(
            daily_box,
            x="date",
            y=y_col2,
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
