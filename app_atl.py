import pandas as pd
import streamlit as st
import altair as alt
from io import BytesIO


# --- Load data ---
df = pd.read_excel("Al_data.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Date'] = df['Date'].dt.date  # YYYY-MM-DD

# Ensure 'Load (kg)' is numeric
df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')

# Check if 'Tempo' exists
has_tempo = 'Tempo' in df.columns
if has_tempo:
    df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')

st.title("🏋️‍♂️ Athletes Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

# Select Area 
areas = df['Area'].dropna().unique()
selected_areas = st.sidebar.multiselect("Select Area(s)", areas)

# Name filter
names = df['Name'].dropna().unique()
selected_names = st.sidebar.multiselect("Select Name(s)", names)

# Year filter
years = sorted(df['Date'].apply(lambda x: x.year).dropna().unique())
selected_year = st.sidebar.multiselect("Select Year(s)", years)

# Filter data for dynamic Month options
df_year_filtered = df.copy()
if selected_year:
    df_year_filtered = df_year_filtered[df_year_filtered['Date'].apply(lambda x: x.year).isin(selected_year)]

# Month filter (dynamic)
months = sorted(df_year_filtered['Date'].apply(lambda x: x.month).dropna().unique())
selected_month = st.sidebar.multiselect("Select Month(s)", months)

# Multi-keyword search filters
code_search = st.sidebar.text_input("Search Code Contains (comma separated)", "")
exercise_search = st.sidebar.text_input("Search Exercise Contains (comma separated)", "")

# Load filter
if 'Load (kg)' in df.columns:
    min_load = float(df['Load (kg)'].min(skipna=True))
    max_load = float(df['Load (kg)'].max(skipna=True))
    load_range = st.sidebar.slider(
        "Select Load (kg) Range", 
        min_value=min_load, 
        max_value=max_load, 
        value=(min_load, max_load)
    )
else:
    load_range = (None, None)

# --- Apply filters ---
filtered_df = df.copy()

if selected_areas:
    filtered_df = filtered_df[filtered_df['Area'].isin(selected_areas)]

if selected_names:
    filtered_df = filtered_df[filtered_df['Name'].isin(selected_names)]

if selected_year:
    filtered_df = filtered_df[filtered_df['Date'].apply(lambda x: x.year).isin(selected_year)]

if selected_month:
    filtered_df = filtered_df[filtered_df['Date'].apply(lambda x: x.month).isin(selected_month)]

# --- Code filter (multi-keyword) ---
if code_search and 'Code' in filtered_df.columns:
    code_keywords = [kw.strip() for kw in code_search.split(",") if kw.strip()]
    if code_keywords:
        filtered_df = filtered_df[filtered_df['Code'].apply(
            lambda x: any(kw.lower() in str(x).lower() for kw in code_keywords)
        )]

# --- Exercise filter (multi-keyword) ---
if exercise_search and 'Exercise' in filtered_df.columns:
    exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]
    if exercise_keywords:
        filtered_df = filtered_df[filtered_df['Exercise'].apply(
            lambda x: any(kw.lower() in str(x).lower() for kw in exercise_keywords)
        )]

# Load filter
if load_range != (None, None):
    filtered_df = filtered_df[filtered_df['Load (kg)'].between(load_range[0], load_range[1])]

# --- Display results ---
st.write("### Filtered Data", filtered_df)

# Summary statistics
summary_cols = ['Set', 'Rep', 'Load (kg)']
if has_tempo:
    summary_cols.append('Tempo')

st.write("### Summary Statistics")
st.dataframe(filtered_df[summary_cols].describe())

# --- Plot Load Distribution Over Time by Keyword ---
if not filtered_df.empty and exercise_search:
    # Prepare keywords
    exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]

    # Assign each row to the first matching keyword
    def match_keyword(ex):
        for kw in exercise_keywords:
            if kw.lower() in str(ex).lower():
                return kw
        return "Other"

    filtered_df['Exercise_Keyword'] = filtered_df['Exercise'].apply(match_keyword)

    # Aggregate average load per Month-Year and keyword
    filtered_df['Month_Year'] = filtered_df['Date'].apply(lambda x: x.strftime('%b-%Y'))
    load_time_df = filtered_df.groupby(['Month_Year', 'Exercise_Keyword'])['Load (kg)'].mean().reset_index()
    load_time_df['Month_Year_Date'] = pd.to_datetime(load_time_df['Month_Year'], format='%b-%Y')
    load_time_df = load_time_df.sort_values('Month_Year_Date')

    # Altair chart with keyword colors
    chart = alt.Chart(load_time_df).mark_bar().encode(
        x=alt.X('Month_Year:N', sort=list(load_time_df['Month_Year']), title='Month-Year'),
        y=alt.Y('Load (kg):Q', title='Average Load (kg)'),
        color=alt.Color('Exercise_Keyword:N', title='Keyword'),
        tooltip=['Month_Year', 'Exercise_Keyword', 'Load (kg)']
    ).properties(
        width=700,
        height=400,
        title="Average Load Distribution Over Time by Filter Keyword"
    )

    st.altair_chart(chart, use_container_width=True)

# --- Pie Chart for Family Column ---
if not filtered_df.empty and 'Family' in filtered_df.columns:
    # Count number of exercises per Family category
    family_counts = filtered_df['Family'].value_counts().reset_index()
    family_counts.columns = ['Family', 'Count']

    # Keep top 5 categories
    top5 = family_counts.head(5)
    others_sum = family_counts['Count'][5:].sum()
    if others_sum > 0:
        # Add "Other" category for remaining
        top5 = pd.concat([top5, pd.DataFrame({'Family': ['Other'], 'Count': [others_sum]})], ignore_index=True)

    # Altair Pie Chart
    pie_chart = alt.Chart(top5).mark_arc().encode(
        theta=alt.Theta(field="Count", type="quantitative"),
        color=alt.Color(field="Family", type="nominal"),
        tooltip=['Family', 'Count']
    ).properties(
        width=400,
        height=400,
        title="Proportion of Exercises by Family (Top 5)"
    )

    st.altair_chart(pie_chart, use_container_width=True)

# --- Download filtered data as CSV ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df(filtered_df)

st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=csv_data,
    file_name="filtered_training.csv",
    mime="text/csv"
)


























