import pandas as pd
import streamlit as st
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np

# --- Load data ---
df = pd.read_excel("All_REHAB.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure 'Date' is datetime and format nicely
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Date'] = df['Date'].dt.date  # YYYY-MM-DD

# Ensure 'Load (kg)' and 'Tempo' are numeric
df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')
df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')  # assuming Tempo column exists

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

# --- Dual-axis bar plot with Matplotlib ---
if not filtered_df.empty:
    # Aggregate by Date
    agg_df = filtered_df.groupby('Date').agg({'Load (kg)': 'sum', 'Tempo': 'mean'}).reset_index()
    
    x = np.arange(len(agg_df['Date']))
    width = 0.4

    fig, ax1 = plt.subplots(figsize=(12,6))

    # Left Y-axis: Load
    ax1.bar(x - width/2, agg_df['Load (kg)'], width=width, color='steelblue', label='Load (kg)')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Load (kg)', color='steelblue')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(agg_df['Date'], rotation=45, ha='right')

    # Right Y-axis: Tempo
    ax2 = ax1.twinx()
    ax2.bar(x + width/2, agg_df['Tempo'], width=width, color='orange', label='Tempo')
    ax2.set_ylabel('Tempo', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    # Legends
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    st.pyplot(fig)

# --- Download filtered data ---
def convert_df_to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Filtered_Data')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

excel_data = convert_df_to_excel(filtered_df)

st.download_button(
    label="📥 Download Filtered Data as Excel",
    data=excel_data,
    file_name="Filtered_All_REHAB.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
