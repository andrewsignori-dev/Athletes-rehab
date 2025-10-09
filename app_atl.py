import streamlit as st
from data import load_data
from filters import apply_filters
from utils import filter_by_keywords
from plots import plot_load_trends

st.set_page_config(page_title="Athletes Dashboard", layout="wide")
st.title("🏋️‍♂️ Athletes Dashboard")

# --- Load Data ---
df = load_data("All_data.xlsx")

# --- Sidebar Filters ---
st.sidebar.header("Filters")
areas = df['Area'].dropna().unique()
selected_areas = st.sidebar.multiselect("Select Area(s)", areas)
names = df['Name'].dropna().unique()
selected_names = st.sidebar.multiselect("Select Name(s)", names)
years = sorted(df['Date'].apply(lambda x: x.year).dropna().unique())
selected_year = st.sidebar.multiselect("Select Year(s)", years)
months = sorted(df['Date'].apply(lambda x: x.month).dropna().unique())
selected_month = st.sidebar.multiselect("Select Month(s)", months)

# Load filter
load_filter_type = st.sidebar.radio("Filter by Load (kg)", ["All", "With Load", "Without Load"])
load_range = None
if load_filter_type == "With Load":
    min_load = float(df['Load (kg)'].min(skipna=True))
    max_load = float(df['Load (kg)'].max(skipna=True))
    load_range = st.sidebar.slider("Select Load (kg) Range", min_value=min_load, max_value=max_load, value=(min_load, max_load))

# Family filter
families = df['Family'].dropna().unique() if 'Family' in df.columns else []
selected_families = st.sidebar.multiselect("Select Family(s)", families)

# Keyword filters
with st.sidebar.expander("🔎 Advanced Filters"):
    code_search = st.text_input("Search Code Contains (comma separated)", "")
    exercise_search = st.text_input("Search Exercise Contains (comma separated)", "")

# --- Apply filters ---
filtered_df = apply_filters(
    df,
    selected_areas,
    selected_names,
    selected_year,
    selected_month,
    load_filter_type,
    load_range,
    selected_families
)
filtered_df = filter_by_keywords(filtered_df, 'Code', [kw.strip() for kw in code_search.split(',') if kw.strip()])
filtered_df = filter_by_keywords(filtered_df, 'Exercise', [kw.strip() for kw in exercise_search.split(',') if kw.strip()])

# --- Metrics ---
col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", f"{len(filtered_df):,}")
col2.metric("Unique Exercises", filtered_df['Exercise'].nunique() if not filtered_df.empty else 0)
latest_date = filtered_df['Date'].max() if not filtered_df.empty else None
col3.metric("Latest Training Date", latest_date.strftime('%d-%m-%Y') if latest_date else "-")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📄 Filtered Data", "📊 Summary Stats", "📈 Trends"])

with tab1:
    st.dataframe(filtered_df)
    if not filtered_df.empty:
        st.download_button("⬇️ Download CSV", filtered_df.to_csv(index=False).encode('utf-8'), "filtered_training.csv", "text/csv")

with tab3:
    exercise_keywords = [kw.strip() for kw in exercise_search.split(',') if kw.strip()]
    fig_bar, heatmap = plot_load_trends(filtered_df, exercise_keywords)
    if fig_bar: st.plotly_chart(fig_bar, use_container_width=True)
    if heatmap: st.plotly_chart(heatmap, use_container_width=True)
