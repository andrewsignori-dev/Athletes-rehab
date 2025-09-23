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
selected_names = st.sidebar.multiselect("Select Name(s)", names, default=names)

# 2️⃣ Date filter
years = sorted(df['Date'].dt.year.dropna().unique())
selected_year = st.sidebar.multiselect("Select Year(s)", years, default=years)

months = sorted(df['Date'].dt.month.dropna().unique())
selected_month = st.sidebar.multiselect("Select Month(s)", months, default=months)

weeks = sorted(df['Date'].dt.isocalendar().week.dropna().unique())
selected_weeks = st.sidebar.multiselect("Select Week(s)", weeks, default=weeks)

# 3️⃣ Load filter
if 'Load (kg)' in df.columns:
    min_load = float(df['Load (kg)'].min(skipna=True))
    max_load = float(df['Load (kg)'].max(skipna=True))
    load_range = st.sidebar.slider("Select Load (kg) Range", min_value=min_load, max_value=max_load, value=(min_load, max_load))
else:
    load_range = (0, 999999)

# 4️⃣ Code filter
code_search = st.sidebar.text_input("Search Code Contains", "")

# --- Apply filters ---
filtered_df = df[
    (df['Name'].isin(selected_names)) &
    (df['Date'].dt.year.isin(selected_year)) &
    (df['Date'].dt.month.isin(selected_month)) &
    (df['Date'].dt.isocalendar().week.isin(selected_weeks)) &
    (df['Load (kg)'].between(load_range[0], load_range[1]))
]

if code_search:
    filtered_df = filtered_df[filtered_df['Code'].str.contains(code_search, case=False, na=False)]

# --- Display results ---
st.write("### Filtered Data", filtered_df)

# --- Summary statistics ---
st.write("### Summary Statistics")
st.dataframe(filtered_df.describe())
