import pandas as pd
import streamlit as st
import plotly.express as px
from io import BytesIO

# --- Load data ---
df = pd.read_excel("All_data.xlsx")

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

st.set_page_config(page_title="Athletes Dashboard", layout="wide")
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
with st.sidebar.expander("🔎 Advanced Filters"):
    code_search = st.text_input("Search Code Contains (comma separated)", "")
    exercise_search = st.text_input("Search Exercise Contains (comma separated)", "")
    if 'Family' in df.columns:
        families = df['Family'].dropna().unique()
        selected_families = st.multiselect("Select Family(s)", families)

# --- Load filter type ---
load_filter_type = st.sidebar.radio(
    "Filter by Load (kg)",
    ("All", "With Load", "Without Load")
)

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

# --- Apply Family filter ---
if 'Family' in filtered_df.columns and 'selected_families' in locals() and selected_families:
    filtered_df = filtered_df[filtered_df['Family'].isin(selected_families)]

# --- Apply load filter ---
if load_filter_type == "With Load":
    filtered_df = filtered_df[filtered_df['Load (kg)'].notna()]
elif load_filter_type == "Without Load":
    filtered_df = filtered_df[filtered_df['Load (kg)'].isna()]
# If "All" do nothing

# Show load range slider only for "With Load"
if load_filter_type == "With Load" and not filtered_df.empty:
    min_load = float(filtered_df['Load (kg)'].min(skipna=True))
    max_load = float(filtered_df['Load (kg)'].max(skipna=True))
    load_range = st.sidebar.slider(
        "Select Load (kg) Range",
        min_value=min_load,
        max_value=max_load,
        value=(min_load, max_load)
    )
    filtered_df = filtered_df[filtered_df['Load (kg)'].between(load_range[0], load_range[1])]

# --- Metrics at the top ---
if not filtered_df.empty:
    latest_date = filtered_df['Date'].max()
else:
    latest_date = pd.NaT

col1, col2, col3 = st.columns(3)
col1.metric("Total Entries", len(filtered_df))
col2.metric("Unique Exercises", filtered_df['Exercise'].nunique() if not filtered_df.empty else 0)
col3.metric("Latest Training Date", latest_date.strftime('%d-%m-%Y') if pd.notna(latest_date) else "-")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["📄 Filtered Data", "📊 Summary Stats", "📈 Trends", "🥧 Family Proportion"])

with tab1:
    st.write("### Filtered Data")
    st.dataframe(filtered_df)  # show filtered instead of full df

    # Last training registered
    if not filtered_df.empty:
        last_training_df = filtered_df[filtered_df['Date'] == latest_date]
        st.write(f"### Last Training Registered (Date: {latest_date.strftime('%d-%m-%Y')})")
        st.dataframe(last_training_df)

    # Download filtered data as CSV
    csv_data = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_data,
        file_name="filtered_training.csv",
        mime="text/csv"
    )

with tab2:
    st.write("### Summary Statistics")
    summary_cols = ['Set', 'Rep', 'Load (kg)']
    if has_tempo:
        summary_cols.append('Tempo')
    if not filtered_df.empty:
        st.dataframe(filtered_df[summary_cols].describe())

with tab3:
    st.write("### Visualize Load Trends Over Time")
    if not filtered_df.empty:
        exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]

        def match_keyword(ex):
            for kw in exercise_keywords:
                if kw.lower() in str(ex).lower():
                    return kw
            return "Other"

        filtered_df['Exercise_Keyword'] = filtered_df['Exercise'].apply(match_keyword)
        filtered_df['Month_Year'] = pd.to_datetime(filtered_df['Date']).dt.to_period('M').astype(str)
        load_time_df = filtered_df.groupby(['Month_Year', 'Exercise_Keyword'])['Load (kg)'].mean().reset_index()

        fig_bar = px.bar(
            load_time_df,
            x='Month_Year',
            y='Load (kg)',
            color='Exercise_Keyword',
            barmode='group',
            title='Average Load (kg) Over Time'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # --- Monthly Boxplot with Median Trendline ---
        filtered_df['Month'] = pd.to_datetime(filtered_df['Date']).dt.to_period('M').dt.to_timestamp()
        fig_box = px.box(
            filtered_df,
            x='Month',
            y='Load (kg)',
            points="outliers",
            color_discrete_sequence=['lightblue'],
            title='Monthly Load Distribution with Median Trendline'
        )

        # Add median trendline manually
        median_df = filtered_df.groupby('Month')['Load (kg)'].median().reset_index()
        fig_box.add_scatter(
            x=median_df['Month'],
            y=median_df['Load (kg)'],
            mode='lines+markers',
            name='Median',
            line=dict(color='red', dash='dash'),
            marker=dict(symbol='circle', size=6)
        )
        fig_box.update_layout(
            xaxis_title="Month",
            yaxis_title="Load (kg)",
            boxmode='group',
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig_box, use_container_width=True)

        # Optional: Heatmap
        heatmap = px.density_heatmap(
            load_time_df,
            x='Month_Year',
            y='Exercise_Keyword',
            z='Load (kg)',
            color_continuous_scale='Blues',
            title='Heatmap of Average Load per Exercise Keyword Over Time'
        )
        st.plotly_chart(heatmap, use_container_width=True)

with tab4:
    st.write("### Proportion of Exercises by Family")
    if not filtered_df.empty and 'Family' in filtered_df.columns:
        family_counts = filtered_df['Family'].value_counts().reset_index()
        family_counts.columns = ['Family', 'Count']
        total = family_counts['Count'].sum()
        family_counts['Percentage'] = (family_counts['Count'] / total * 100).round(1)

        fig_pie = px.pie(
            family_counts,
            names='Family',
            values='Count',
            title='Proportion of Exercises by Family',
            hole=0.3
        )

        # Make slices "pop out" a bit
        fig_pie.update_traces(textinfo='label+percent', pull=[0.05]*len(family_counts))

        # Increase overall figure size
        fig_pie.update_layout(
            width=700,
            height=700,
            legend=dict(
                title="Family (with %)",
                orientation="v",  # vertical
                x=1.05,  # move legend outside
                y=0.5
            ),
            title=dict(font=dict(size=20))
        )

        st.plotly_chart(fig_pie)

       # --- Table with weekly stats ---
        filtered_df['Week'] = pd.to_datetime(filtered_df['Date']).dt.isocalendar().week

        # Determine grouping columns
        group_cols = ['Family', 'Area']
        show_exercise = 'selected_families' in locals() and selected_families
        if show_exercise:
            group_cols.append('Exercise')

        # Aggregate including Week_From and Week_To
        agg_df = filtered_df.groupby(group_cols).agg(
            Week_From=('Week', 'min'),
            Week_To=('Week', 'max'),
            Avg_Load=('Load (kg)', 'mean')
        ).reset_index()

        # Pivot to separate Rehab and S&C
        if show_exercise:
            pivot_df = agg_df.pivot_table(
                index=['Family', 'Exercise', 'Week_From', 'Week_To'],
                columns='Area',
                values='Avg_Load'
            ).reset_index()
        else:
            pivot_df = agg_df.pivot_table(
                index=['Family', 'Week_From', 'Week_To'],
                columns='Area',
                values='Avg_Load'
            ).reset_index()

        pivot_df.columns.name = None  # remove hierarchy

        # Ensure Avg_Load columns always exist
        for col in ['Rehab', 'S&C']:
            if col not in pivot_df.columns:
                pivot_df[col] = 0
            else:
                # Convert to numeric safely and fill missing with 0
                pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce').fillna(0)


        pivot_df.rename(columns={'Rehab': 'Avg_Load_Rehab', 'S&C': 'Avg_Load_S&C'}, inplace=True)

        # Round Avg_Load columns
        for col in ['Avg_Load_Rehab', 'Avg_Load_S&C']:
            pivot_df[col] = pivot_df[col].round(1)

        # Display table
        st.write("### Weekly Load Summary")
        st.dataframe(pivot_df)


















































