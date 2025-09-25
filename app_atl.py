import pandas as pd
import streamlit as st
from io import BytesIO
import altair as alt

# --- Load data ---
df = pd.read_excel("All_REHAB.xlsx")

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

# --- Plot Load Distribution over Time by Exercise ---
if not filtered_df.empty:
    filtered_df['Month_Year'] = filtered_df['Date'].apply(lambda x: x.strftime('%b-%Y'))

    # Decide if we have multiple exercises in the filter
    if exercise_search:
        exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]
        if len(exercise_keywords) > 1:
            # Aggregate average load per Month-Year and Exercise
            load_time_df = filtered_df.groupby(['Month_Year', 'Exercise'])['Load (kg)'].mean().reset_index()
            load_time_df['Month_Year_Date'] = pd.to_datetime(load_time_df['Month_Year'], format='%b-%Y')
            load_time_df = load_time_df.sort_values('Month_Year_Date')

            chart = alt.Chart(load_time_df).mark_bar().encode(
                x=alt.X('Month_Year:N', sort=list(load_time_df['Month_Year']), title='Month-Year'),
                y=alt.Y('Load (kg):Q', title='Average Load (kg)'),
                color=alt.Color('Exercise:N', title='Exercise'),
                tooltip=['Month_Year', 'Exercise', 'Load (kg)']
            ).properties(
                width=700,
                height=400,
                title="Average Load Distribution Over Time by Exercise"
            )
        else:
            # Single exercise selected, normal chart
            load_time_df = filtered_df.groupby('Month_Year')['Load (kg)'].mean().reset_index()
            load_time_df['Month_Year_Date'] = pd.to_datetime(load_time_df['Month_Year'], format='%b-%Y')
            load_time_df = load_time_df.sort_values('Month_Year_Date')

            chart = alt.Chart(load_time_df).mark_bar(color='skyblue').encode(
                x=alt.X('Month_Year:N', sort=list(load_time_df['Month_Year']), title='Month-Year'),
                y=alt.Y('Load (kg):Q', title='Average Load (kg)'),
                tooltip=['Month_Year', 'Load (kg)']
            ).properties(
                width=700,
                height=400,
                title="Average Load Distribution Over Time"
            )
    else:
        # No exercise filter, normal chart
        load_time_df = filtered_df.groupby('Month_Year')['Load (kg)'].mean().reset_index()
        load_time_df['Month_Year_Date'] = pd.to_datetime(load_time_df['Month_Year'], format='%b-%Y')
        load_time_df = load_time_df.sort_values('Month_Year_Date')

        chart = alt.Chart(load_time_df).mark_bar(color='skyblue').encode(
            x=alt.X('Month_Year:N', sort=list(load_time_df['Month_Year']), title='Month-Year'),
            y=alt.Y('Load (kg):Q', title='Average Load (kg)'),
            tooltip=['Month_Year', 'Load (kg)']
        ).properties(
            width=700,
            height=400,
            title="Average Load Distribution Over Time"
        )

    st.altair_chart(chart, use_container_width=True)

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





















