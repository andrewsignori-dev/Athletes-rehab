import pandas as pd
import streamlit as st
import altair as alt
from io import BytesIO
from fpdf import FPDF
import altair_saver

# --- Load data ---
df = pd.read_excel("All_REHAB.xlsx")
df.columns = df.columns.str.strip()

# Ensure 'Date' is datetime
df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')

has_tempo = 'Tempo' in df.columns
if has_tempo:
    df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')

st.title("🏋️‍♂️ Athletes Dashboard")

# --- Sidebar filters ---
st.sidebar.header("Filters")

areas = df['Area'].dropna().unique()
selected_areas = st.sidebar.multiselect("Select Area(s)", areas)

names = df['Name'].dropna().unique()
selected_names = st.sidebar.multiselect("Select Name(s)", names)

years = sorted(df['Date'].apply(lambda x: x.year).dropna().unique())
selected_year = st.sidebar.multiselect("Select Year(s)", years)

df_year_filtered = df.copy()
if selected_year:
    df_year_filtered = df_year_filtered[df_year_filtered['Date'].apply(lambda x: x.year).isin(selected_year)]

months = sorted(df_year_filtered['Date'].apply(lambda x: x.month).dropna().unique())
selected_month = st.sidebar.multiselect("Select Month(s)", months)

code_search = st.sidebar.text_input("Search Code Contains (comma separated)", "")
exercise_search = st.sidebar.text_input("Search Exercise Contains (comma separated)", "")

if 'Load (kg)' in df.columns:
    min_load = float(df['Load (kg)'].min(skipna=True))
    max_load = float(df['Load (kg)'].max(skipna=True))
    load_range = st.sidebar.slider(
        "Select Load (kg) Range", min_value=min_load, max_value=max_load, value=(min_load, max_load)
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
if code_search and 'Code' in filtered_df.columns:
    code_keywords = [kw.strip() for kw in code_search.split(",") if kw.strip()]
    if code_keywords:
        filtered_df = filtered_df[filtered_df['Code'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in code_keywords))]
if exercise_search and 'Exercise' in filtered_df.columns:
    exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]
    if exercise_keywords:
        filtered_df = filtered_df[filtered_df['Exercise'].apply(lambda x: any(kw.lower() in str(x).lower() for kw in exercise_keywords))]
if load_range != (None, None):
    filtered_df = filtered_df[filtered_df['Load (kg)'].between(load_range[0], load_range[1])]

# --- Display filtered data and summary ---
st.write("### Filtered Data", filtered_df)
summary_cols = ['Set', 'Rep', 'Load (kg)']
if has_tempo:
    summary_cols.append('Tempo')

st.write("### Summary Statistics")
st.dataframe(filtered_df[summary_cols].describe())

# --- Plot Load Distribution by Exercise Keyword ---
chart = None
if not filtered_df.empty and exercise_search:
    exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]

    def match_keyword(ex):
        for kw in exercise_keywords:
            if kw.lower() in str(ex).lower():
                return kw
        return "Other"

    filtered_df['Exercise_Keyword'] = filtered_df['Exercise'].apply(match_keyword)
    filtered_df['Month_Year'] = filtered_df['Date'].apply(lambda x: x.strftime('%b-%Y'))

    load_time_df = filtered_df.groupby(['Month_Year', 'Exercise_Keyword'])['Load (kg)'].mean().reset_index()
    load_time_df['Month_Year_Date'] = pd.to_datetime(load_time_df['Month_Year'], format='%b-%Y')
    load_time_df = load_time_df.sort_values('Month_Year_Date')

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

# --- CSV download ---
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="⬇️ Download Filtered Data as CSV",
    data=convert_df(filtered_df),
    file_name="filtered_training.csv",
    mime="text/csv"
)

# --- PDF download ---
def altair_chart_to_png(chart):
    buf = BytesIO()
    altair_saver.save(chart, buf, fmt="png")
    buf.seek(0)
    return buf

def generate_pdf(filtered_df, chart, filters_summary):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Athletes Dashboard Report", ln=True, align="C")

    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.multi_cell(0, 8, f"Filters Applied:\n{filters_summary}")
    pdf.ln(5)
    pdf.cell(0, 8, f"Total Rows: {len(filtered_df)}", ln=True)

    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 8, "Summary Statistics", ln=True)
    pdf.set_font("Arial", '', 10)
    summary_df = filtered_df[summary_cols].describe()
    for i, row in summary_df.iterrows():
        pdf.multi_cell(0, 6, f"{i}: {row.to_dict()}")

    if chart:
        chart_png = altair_chart_to_png(chart)
        pdf.image(chart_png, x=10, y=None, w=180)

    output = BytesIO()
    pdf.output(output)
    output.seek(0)
    return output

# Build filters summary text
filters_summary = ""
if selected_areas:
    filters_summary += f"Areas: {', '.join(selected_areas)}\n"
if selected_names:
    filters_summary += f"Names: {', '.join(selected_names)}\n"
if selected_year:
    filters_summary += f"Years: {', '.join(map(str, selected_year))}\n"
if selected_month:
    filters_summary += f"Months: {', '.join(map(str, selected_month))}\n"
if code_search:
    filters_summary += f"Code Keywords: {code_search}\n"
if exercise_search:
    filters_summary += f"Exercise Keywords: {exercise_search}\n"
filters_summary += f"Load Range: {load_range[0]} - {load_range[1]} kg\n"

# Show PDF download button
if not filtered_df.empty:
    pdf_buffer = generate_pdf(filtered_df, chart, filters_summary)
    st.download_button(
        label="⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="athletes_report.pdf",
        mime="application/pdf"
    )
























