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

# --- Dual-axis chart with Altair ---
if not filtered_df.empty:
    agg_df = filtered_df.groupby('Date').agg({'Load (kg)': 'sum', 'Tempo': 'mean'}).reset_index()

    base = alt.Chart(agg_df).encode(x=alt.X('Date:T', title='Date'))

    load_bar = base.mark_bar(color='steelblue').encode(
        y=alt.Y('Load (kg):Q', axis=alt.Axis(title='Load (kg)'))
    )

    tempo_bar = base.mark_bar(color='orange').encode(
        y=alt.Y('Tempo:Q', axis=alt.Axis(title='Tempo'))
    )

    # Combine with second y-axis
    chart = alt.layer(
        load_bar,
        tempo_bar
    ).resolve_scale(
        y='independent'
    ).properties(
        width=800,
        height=400,
        title="Load and Tempo per Date"
    )

    st.altair_chart(chart, use_container_width=True)

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
