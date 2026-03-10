import streamlit as st
import pandas as pd
import altair as alt


@st.cache_data
def load_data(path: str = "data/synthetic_conversations.csv"):
    return pd.read_csv(path, parse_dates=["created_at"])


def main():
    st.title("AI Assistance — Causal Analysis Dashboard")
    st.write("Simple dashboard showing treatment vs control outcomes.")

    df = load_data()
    st.sidebar.markdown("## Filters")
    min_sev, max_sev = int(df.issue_severity.min()), int(df.issue_severity.max())
    sev = st.sidebar.slider("Issue severity", min_value=min_sev, max_value=max_sev, value=(min_sev, max_sev))
    df = df[(df.issue_severity >= sev[0]) & (df.issue_severity <= sev[1])]

    st.header("Outcome summaries")
    summary = df.groupby("ai_assisted")["resolution_time", "satisfaction_score", "escalated"].mean()
    st.dataframe(summary.round(3))

    st.header("Resolution time distribution")
    chart = alt.Chart(df).transform_density(
        "resolution_time",
        as_=["resolution_time", "density"],
        groupby=["ai_assisted"],
    ).mark_area(opacity=0.5).encode(
        x="resolution_time:Q",
        y="density:Q",
        color=alt.Color("ai_assisted:N", title="AI assisted"),
    )
    st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    main()
