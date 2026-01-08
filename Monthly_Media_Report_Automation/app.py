import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from streamlit_plotly_events import plotly_events

# ----------------------------------------------------
# Page configuration
# ----------------------------------------------------
st.set_page_config(page_title="Media Intelligence Dashboard", layout="wide")
st.title("📊 Media Intelligence Dashboard")

# ----------------------------------------------------
# Sidebar controls
# ----------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader("Upload Meltwater CSV", type="csv")

time_view = st.sidebar.radio("Time aggregation", ["Daily", "Weekly", "Monthly"], index=1)
brand_filter = st.sidebar.text_input("Brand / topic filter (headline)", help="Comma-separated keywords")
wordcloud_source = st.sidebar.radio("Word cloud source", ["Headline", "Keyword"], index=0)

# ----------------------------------------------------
# Utility functions
# ----------------------------------------------------
def read_csv_safe(file):
    import chardet
    raw = file.read()
    encoding = chardet.detect(raw).get("encoding", "latin-1")
    file.seek(0)
    for sep in [",", ";", "\t"]:
        try:
            df = pd.read_csv(file, encoding=encoding, sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            file.seek(0)
    st.error("Unable to parse CSV file.")
    st.stop()

def detect_top_spikes(df, agg_col, top_n=5):
    agg = df.groupby(agg_col).size().reset_index(name="Mentions")
    top = agg.nlargest(top_n, "Mentions")
    dominant_sentiments = []
    for period in top[agg_col]:
        subset = df[df[agg_col]==period]
        dominant = subset["sentiment"].value_counts().idxmax()
        dominant_sentiments.append(dominant)
    top["DominantSentiment"] = dominant_sentiments
    return top

def generate_hover_text(df, agg_col, max_headlines=5):
    hover_texts = []
    grouped = df.groupby(agg_col)
    for period, group in grouped:
        headlines = []
        for _, row in group.head(max_headlines).iterrows():
            sentiment_icon = "🟢" if str(row["sentiment"]).lower()=="positive" else ("🔴" if str(row["sentiment"]).lower()=="negative" else "🟡")
            headline_text = row["headline"][:120]+"..." if len(row["headline"])>120 else row["headline"]
            headlines.append(f"{sentiment_icon} {headline_text}")
        hover_texts.append("<br>".join(headlines))
    return hover_texts

# ----------------------------------------------------
# Main logic
# ----------------------------------------------------
if uploaded_file:
    df = read_csv_safe(uploaded_file)

    # Normalize columns
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    column_map = {
        "date": "date",
        "headline": "headline",
        "title": "headline",
        "source": "source_name",
        "source_name": "source_name",
        "sentiment": "sentiment",
        "keyword": "keyword",
        "keywords": "keyword"  # <-- mapping your CSV's Keywords column
    }
    df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

    # Required columns
    required = ["date", "headline", "source_name", "sentiment"]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}")
            st.stop()

    # Optional keyword column
    if "keyword" not in df.columns:
        df["keyword"] = None

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    # Brand/topic filter
    if brand_filter:
        keywords = [k.strip().lower() for k in brand_filter.split(",")]
        pattern = "|".join(keywords)
        df = df[df["headline"].str.lower().str.contains(pattern, na=False)]

    # Time columns
    df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
    df["month"] = df["date"].dt.to_period("M").astype(str)

    # Date range filter
    min_date, max_date = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Select date range", [min_date, max_date])
    if len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df[(df["date"]>=start_date) & (df["date"]<=end_date)]

    # Time aggregation
    agg_col = "date" if time_view=="Daily" else "week" if time_view=="Weekly" else "month"
    trend_df = df.groupby(agg_col).size().reset_index(name="Mentions").sort_values(agg_col)
    trend_df["DominantSentiment"] = df.groupby(agg_col)["sentiment"].agg(lambda x: x.value_counts().idxmax()).values

    # Generate hover text for trend chart
    trend_hover = generate_hover_text(df, agg_col, max_headlines=5)

    # Clickable trend chart
    st.subheader(f"{time_view} Coverage Trend")
    trend_fig = px.line(trend_df, x=agg_col, y="Mentions", markers=True, hover_data={"Mentions":True})
    for i, trace in enumerate(trend_fig.data):
        trace.hovertemplate = "%{x}<br>Mentions: %{y}<br>Top Headlines:<br>" + "%{customdata[0]}"
        trace.customdata = [[txt] for txt in trend_hover]

    # Highlight peak
    peak_idx = trend_df["Mentions"].idxmax()
    trend_fig.add_scatter(
        x=[trend_df[agg_col].iloc[peak_idx]],
        y=[trend_df["Mentions"].iloc[peak_idx]],
        mode="markers",
        marker=dict(size=14, color="red"),
        name="Peak"
    )

    selected_points = plotly_events(trend_fig, click_event=True)
    if selected_points:
        selected_index = selected_points[0]["pointIndex"]
        selected_period = trend_df.iloc[selected_index][agg_col]
    else:
        selected_period = trend_df.iloc[peak_idx][agg_col]

    period_df = df[df[agg_col] == selected_period]
    st.plotly_chart(trend_fig, use_container_width=True)

    # ------------------------------------------------
    # KPI Row
    # ------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Mentions", len(df))
    col2.metric("Selected Period", str(selected_period))
    col3.metric("Mentions in Period", len(period_df))
    # MoM change
    monthly = df.groupby("month").size().reset_index(name="Mentions").sort_values("month")
    mom_change = ((monthly["Mentions"].iloc[-1]-monthly["Mentions"].iloc[-2])/monthly["Mentions"].iloc[-2]*100) if len(monthly)>1 else 0
    col4.metric("MoM Change", f"{mom_change:.2f}%")

    # ------------------------------------------------
    # Top sources
    # ------------------------------------------------
    st.subheader("Top Sources by Mentions")
    top_sources = df.groupby("source_name").size().reset_index(name="Mentions").sort_values("Mentions", ascending=False).head(5)
    st.dataframe(top_sources, use_container_width=True)

    # ------------------------------------------------
    # News contributing to selected period
    # ------------------------------------------------
    st.subheader("News Contributing to Selected Period")
    st.dataframe(period_df[["date","source_name","headline","sentiment"]], use_container_width=True)

    # ------------------------------------------------
    # Sentiment trend
    # ------------------------------------------------
    st.subheader("Sentiment Distribution")
    sentiment_counts = period_df["sentiment"].value_counts().reindex(["Positive","Negative","Neutral"], fill_value=0)
    sentiment_fig = px.bar(
        x=sentiment_counts.index,
        y=sentiment_counts.values,
        color=sentiment_counts.index,
        labels={"x":"Sentiment","y":"Mentions"},
        title="Sentiment Distribution"
    )
    st.plotly_chart(sentiment_fig, use_container_width=True)

    # ------------------------------------------------
    # Word cloud
    # ------------------------------------------------
    st.subheader("Key Themes")
    if wordcloud_source == "Keyword" and period_df["keyword"].notna().any():
        text_source = period_df["keyword"]
    else:
        text_source = period_df["headline"]
        if wordcloud_source=="Keyword":
            st.warning("Keyword column missing; falling back to headlines for word cloud.")

    text = " ".join(text_source.dropna().astype(str))
    if text.strip():
        wc = WordCloud(width=900, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(12,4))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # ------------------------------------------------
    # Coverage spike analysis
    # ------------------------------------------------
    st.subheader("Top Coverage Spikes")
    top_spikes_df = detect_top_spikes(df, agg_col, top_n=5)
    top_spikes_df["hover"] = generate_hover_text(df[df[agg_col].isin(top_spikes_df[agg_col])], agg_col)
    spike_fig = px.bar(
        top_spikes_df,
        x=agg_col,
        y="Mentions",
        color="DominantSentiment",
        text="Mentions",
        hover_data={"Mentions":True,"DominantSentiment":True,"hover":True}
    )
    spike_fig.update_traces(hovertemplate="%{x}<br>Mentions: %{y}<br>Dominant Sentiment: %{customdata[1]}<br>Top Headlines:<br>%{customdata[2]}")
    st.plotly_chart(spike_fig, use_container_width=True)

    # ------------------------------------------------
    # Narrative insights
    # ------------------------------------------------
    st.subheader("Automatic Narrative Insights")
    insight = ""
    peak_sentiment = top_spikes_df["DominantSentiment"].iloc[0]
    if peak_sentiment.lower()=="negative":
        insight += f"⚠️ Coverage spike driven by negative sentiment in {time_view.lower()} period {top_spikes_df[agg_col].iloc[0]}.\n"
    elif peak_sentiment.lower()=="positive":
        insight += f"🎉 Coverage spike driven by positive sentiment in {time_view.lower()} period {top_spikes_df[agg_col].iloc[0]}.\n"
    else:
        insight += f"🟡 Coverage spike driven by neutral sentiment in {time_view.lower()} period {top_spikes_df[agg_col].iloc[0]}.\n"
    st.markdown(insight)
