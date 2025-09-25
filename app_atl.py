import pandas as pd
import streamlit as st
from io import BytesIO
import altair as alt

# --- Load data ---
df = pd.read_excel("All_REHAB.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure 'Date' is datetime and format nicely
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Date'] = df['Date'].dt.date  # YYYY-MM-DD

# Ensure 'Load (kg)' is numeric
df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')

# Check if 'Tempo' exists
has_tempo = 'Tempo' in df.columns
if has_tempo:
    df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')

st.title("🏋️‍♂️ All_REHAB Dashboard")

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

# Code filter
code_search = st.sidebar.text_input("Search Code Contains", "")
exercise_search = st.sidebar.text_input("Search Exercise Contains", "")

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

if code_search:
    filtered_df = filtered_df[filtered_df['Code'].str.contains(code_search, case=False, na=False)]

if exercise_search:
    filtered_df = filtered_df[filtered_df['Exercise'].str.contains(exercise_search, case=False, na=False)]

if load_range != (None, None):
    filtered_df = filtered_df[filtered_df['Load (kg)'].between(load_range[0], load_range[1])]

# --- Display results ---
st.write("### Filtered Data", filtered_df)
# Select only columns for summary statistics
summary_cols = ['Set', 'Rep', 'Load (kg)']
if 'Tempo' in filtered_df.columns:
    summary_cols.append('Tempo')

st.write("### Summary Statistics")
st.dataframe(filtered_df[summary_cols].describe())

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










