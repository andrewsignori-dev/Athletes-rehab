import pandas as pd
import streamlit as st

# --- Load data ---
df = pd.read_excel("All_REHAB.xlsx")

# Clean column names
df.columns = df.columns.str.strip()

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Ensure 'Load (kg)' is numeric
df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')

st.title("🏋️‍♂️ All_REHAB Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

# 1️⃣ Name filter
names = df['Name'].dropna().unique()
selected_names = st.sidebar.multiselect("Select Name(s)", names)

# 2️⃣ Year filter
years = sorted(df['Date'].dt.year.dropna().unique())
selected_year = st.sidebar.multiselect("Select Year(s)", years)

# Filter data for dynamic Month and Week options
df_year_filtered = df.copy()
if selected_year:
    df_year_filtered = df_year_filtered[df_year_filtered['Date'].dt.year.isin(selected_year)]

# 3️⃣ Month filter (dynamic)
months = sorted(df_year_filtered['Date'].dt.month.dropna().unique())
selected_month = st.sidebar.multiselect("Select Month(s)", months)

# Further filter for dynamic Week options
df_month_filtered = df_year_filtered.copy()
if selected_month:
    df_month_filtered = df_month_filtered[df_month_filtered['Date'].dt.month.isin(selected_month)]

# 4️⃣ Week filter (dynamic)
weeks = sorted(df_month_filtered['Date'].dt.isocalendar().week.dropna().unique())
selected_weeks = st.sidebar.multiselect("Select Week(s)", weeks)

# 5️⃣ Load filter
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

# 6️⃣ Code filter
code_search = st.sidebar.text_input("Search Code Contains", "")

# --- Apply filters only if selected ---
filtered_df = df.copy()

if selected_names:
    filtered_df = filtered_df[filtered_df['Name'].isin(selected_names)]

if selected_year:
    filtered_df = filtered_df[filtered_df['Date'].dt.year.isin(selected_year)]

if selected_month:
    filtered_df = filtered_df[filtered_df['Date'].dt.month.isin(selected_month)]

if selected_weeks:
    filtered_df = filtered_df[filtered_df['Date'].dt.isocalendar().week.isin(selected_weeks)]

if load_range != (None, None):
    filtered_df = filtered_df[filtered_df['Load (kg)'].between(load_range[0], load_range[1])]

if code_search:
    filtered_df = filtered_df[filtered_df['Code'].str.contains(code_search, case=False, na=False)]

# --- Display results ---
st.write("### Filtered Data", filtered_df)

# --- Summary statistics ---
st.write("### Summary Statistics")
st.dataframe(filtered_df.describe())
