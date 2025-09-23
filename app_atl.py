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

# Code filter
code_search = st.sidebar.text_input("Search Code Contains", "")

# --- Apply filters ---
filtered_df = df.copy()

if selected_names:
    filtered_df = filtered_df[filtered_df['Name'].isin(selected_names)]

if selected_year:
    filtered_df = filtered_df[filtered_df['Date'].apply(lambda x: x.year).isin(selected_year)]

if selected_month:
    filtered_df = filtered_df[filtered_df['Date'].apply(lambda x: x.month).isin(selected_month)]

if load_range != (None, None):
    filtered_df = filtered_df[filtered_df['Load (kg)'].between(load_range[0], load_range[1])]

if code_search:
    filtered_df = filtered_df[filtered_df['Code'].str.contains(code_search, case=False, na=False)]

# --- Display results ---
st.write("### Filtered Data", filtered_df)
st.write("### Summary Statistics")
st.dataframe(filtered_df.describe())

# --- Altair chart ---
if not filtered_df.empty:
    agg_df = filtered_df.groupby('Date').agg({'Load (kg)': 'sum'}).reset_index()
    base = alt.Chart(agg_df).encode(x=alt.X('Date:T', title='Date'))

    load_bar = base.mark_bar(color='steelblue').encode(
        y=alt.Y('Load (kg):Q', axis=alt.Axis(title='Load (kg)'))
    )

    if has_tempo:
        tempo_df = filtered_df.groupby('Date').agg({'Tempo':'mean'}).reset_index()
        tempo_line = alt.Chart(tempo_df).mark_line(color='orange', size=3).encode(
            x='Date:T',
            y=alt.Y('Tempo:Q', axis=alt.Axis(title='Tempo'))
        )
        chart = alt.layer(load_bar, tempo_line).resolve_scale(y='independent').properties(
            width=800, height=400, title='Load and Tempo per Date'
        )
    else:
        chart = load_bar.properties(width=800, height=400, title='Load per Date')

    st.altair_chart(chart, use_container_width=True)

# -----------------------------
# Download filtered data
# -----------------------------
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_filtered)
st.download_button("⬇️ Download filtered data", csv, "filtered_training.csv", "text/csv")

