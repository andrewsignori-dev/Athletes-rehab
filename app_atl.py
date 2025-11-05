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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📄 Filtered Data", "📊 Summary Stats", "📈 Trends", "🥧 Family Proportion", "🏆 Competition Analyser", "🏆 Competition Predictor - S&C"])

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
    summary_cols = ['Set', 'Rep', 'Load (kg)', 'Tempo (seconds)']
    if has_tempo:
        summary_cols.append('Tempo')
    if not filtered_df.empty:
        st.dataframe(filtered_df[summary_cols].describe())

 # --- Weekly Load Summary by Exercise Keyword ---
    if exercise_search and 'Exercise' in filtered_df.columns:
        exercise_keywords = [kw.strip() for kw in exercise_search.split(",") if kw.strip()]
        if exercise_keywords:

            # Filter exercises containing the keyword
            matched_df = filtered_df[filtered_df['Exercise'].apply(
                lambda x: any(kw.lower() in str(x).lower() for kw in exercise_keywords)
            )].copy()

            if not matched_df.empty:
                # Week number and year
                matched_df['Week'] = pd.to_datetime(matched_df['Date']).dt.isocalendar().week
                matched_df['Year'] = pd.to_datetime(matched_df['Date']).dt.isocalendar().year

                # Aggregate: min/max week and average load per Exercise & Area
                group_cols = ['Exercise', 'Area']
                agg_df = matched_df.groupby(group_cols).agg(
                    Week_From_Num=('Week', 'min'),
                    Week_To_Num=('Week', 'max'),
                    Year=('Year', 'first'),
                    Avg_Load=('Load (kg)', 'mean')
                ).reset_index()

                # Functions to convert week number to Monday/Sunday
                def week_start_date(week, year):
                    return pd.to_datetime(f'{year}-W{int(week)}-1', format='%G-W%V-%u')
                def week_end_date(week, year):
                    return pd.to_datetime(f'{year}-W{int(week)}-7', format='%G-W%V-%u')

                # Convert week numbers to actual dates
                agg_df['Week_From'] = agg_df.apply(lambda x: week_start_date(x['Week_From_Num'], x['Year']).strftime('%d-%m-%y'), axis=1)
                agg_df['Week_To'] = agg_df.apply(lambda x: week_end_date(x['Week_To_Num'], x['Year']).strftime('%d-%m-%y'), axis=1)
                agg_df.drop(columns=['Week_From_Num', 'Week_To_Num', 'Year'], inplace=True)

                # Pivot Rehab vs S&C
                pivot_df = agg_df.pivot_table(
                    index=['Exercise', 'Week_From', 'Week_To'],
                    columns='Area',
                    values='Avg_Load',
                    fill_value=0
                ).reset_index()

                pivot_df.columns.name = None

                # Ensure columns exist and are numeric, fill missing with 0
                for col in ['Rehabilitation', 'S&C']:
                    if col not in pivot_df.columns:
                        pivot_df[col] = 0
                    else:
                        pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce').fillna(0)

                # Rename for clarity
                pivot_df.rename(columns={'Rehabilitation': 'Avg_Load_Rehab', 'S&C': 'Avg_Load_S&C'}, inplace=True)

                # Round numeric columns
                pivot_df['Avg_Load_Rehab'] = pivot_df['Avg_Load_Rehab'].round(1)
                pivot_df['Avg_Load_S&C'] = pivot_df['Avg_Load_S&C'].round(1)

                # Sort table
                pivot_df = pivot_df.sort_values(by=['Exercise', 'Week_From']).reset_index(drop=True)

                st.write(f"### Load Summary for Exercises containing: {', '.join(exercise_keywords)}")
                st.dataframe(pivot_df)



with tab3:
    st.write("### Visualize Mean Load Trends Over Time")
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
            barmode='relative',
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

        # --- Weekly Load Summary Table ---
        filtered_df['Week'] = pd.to_datetime(filtered_df['Date']).dt.isocalendar().week
        filtered_df['Year'] = pd.to_datetime(filtered_df['Date']).dt.isocalendar().year

        group_cols = ['Family', 'Area']
        show_exercise = 'selected_families' in locals() and selected_families
        if show_exercise:
            group_cols.append('Exercise')

        # Aggregate: min/max week and average load
        agg_df = filtered_df.groupby(group_cols).agg(
            Week_From_Num=('Week', 'min'),
            Week_To_Num=('Week', 'max'),
            Year=('Year', 'first'),
            Avg_Load=('Load (kg)', 'mean')
        ).reset_index()

        # Functions to convert week number to Monday/Sunday
        def week_start_date(week, year):
            return pd.to_datetime(f'{year}-W{int(week)}-1', format='%G-W%V-%u')
        def week_end_date(week, year):
            return pd.to_datetime(f'{year}-W{int(week)}-7', format='%G-W%V-%u')

        # Convert week numbers to actual dates
        agg_df['Week_From'] = agg_df.apply(lambda x: week_start_date(x['Week_From_Num'], x['Year']).strftime('%d-%m-%y'), axis=1)
        agg_df['Week_To'] = agg_df.apply(lambda x: week_end_date(x['Week_To_Num'], x['Year']).strftime('%d-%m-%y'), axis=1)
        agg_df.drop(columns=['Week_From_Num', 'Week_To_Num', 'Year'], inplace=True)

        # Pivot Rehab vs S&C
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

        pivot_df.columns.name = None

        # Ensure columns exist and are numeric, fill missing with 0
        for col in ['Rehabilitation', 'S&C']:
            if col not in pivot_df.columns:
                pivot_df[col] = 0
            else:
                pivot_df[col] = pd.to_numeric(pivot_df[col], errors='coerce').fillna(0)

        pivot_df.rename(columns={'Rehabilitation': 'Avg_Load_Rehab', 'S&C': 'Avg_Load_S&C'}, inplace=True)

        # Round numeric columns
        pivot_df['Avg_Load_Rehab'] = pivot_df['Avg_Load_Rehab'].round(1)
        pivot_df['Avg_Load_S&C'] = pivot_df['Avg_Load_S&C'].round(1)

        # Optional: sort table for readability
        sort_cols = ['Family', 'Week_From']
        if show_exercise:
            sort_cols.append('Exercise')
        pivot_df = pivot_df.sort_values(by=sort_cols).reset_index(drop=True)

        st.write("### Load Summary by Family")
        st.dataframe(pivot_df)

with tab5:
    st.write("### 🏆 Competition Analyser")

    # --- Check necessary columns ---
    required_cols = ['Name', 'Date', 'Competition (positioning)', 'Area', 'Set', 'Rep', 'Load (kg)']
    if not all(col in filtered_df.columns for col in required_cols):
        st.warning("Missing one or more required columns: Name, Date, Competition (positioning), Area, Set, Rep, Load")
    elif filtered_df.empty:
        st.info("No data available for competition analysis. Please adjust your filters.")
    else:
        # --- Filters for Name, Year, and Area ---
        st.subheader("Filters")

        # Filter by Athlete Name
        available_names = sorted(filtered_df['Name'].dropna().unique())
        selected_name = st.selectbox(
            "Select Athlete", 
            available_names, 
            key="competition_name_select"
        )

        # Filter by Year
        df_name_filtered = filtered_df[filtered_df['Name'] == selected_name]
        available_years = sorted(pd.to_datetime(df_name_filtered['Date']).dt.year.dropna().unique())
        selected_years = st.multiselect(
            "Select Year(s)", 
            available_years, 
            default=available_years[-1:], 
            key="competition_year_select"
        )

        # --- Filter Data ---
        df_selected = df_name_filtered[
            pd.to_datetime(df_name_filtered['Date']).dt.year.isin(selected_years)
        ].copy()
        
        if df_selected.empty:
            st.info("No competition records found for this athlete and selected year(s).")
        else:
            # Ensure numeric Competition Positioning
            df_selected['Competition (positioning)'] = pd.to_numeric(
                df_selected['Competition (positioning)'], errors='coerce'
            )

            # Clean data
            df_selected = df_selected.dropna(subset=['Competition (positioning)'])
            df_selected['Date'] = pd.to_datetime(df_selected['Date']).dt.date

            # --- Deduplicate competitions (keep one row per competition per date) ---
            df_display = (
                df_selected[['Name', 'Date', 'Competition (positioning)']]
                .drop_duplicates(subset=['Name', 'Date', 'Competition (positioning)'])
                .sort_values(by='Date')
            )

            if df_display.empty:
                st.info("No valid competition positioning data available.")
            else:
                # --- Identify Best and Worst ---
                best_row = df_display.loc[df_display['Competition (positioning)'].idxmin()]
                worst_row = df_display.loc[df_display['Competition (positioning)'].idxmax()]

                st.markdown(f"""
                #### 🥇 Best Performance  
                **Date:** {best_row['Date']}  
                **Competition Positioning:** {int(best_row['Competition (positioning)'])}
                """)

                st.markdown(f"""
                #### 🥈 Worst Performance  
                **Date:** {worst_row['Date']}  
                **Competition Positioning:** {int(worst_row['Competition (positioning)'])}
                """)

                # --- Display Filtered Dataset ---
                st.write("### 📋 Competition Results Table")
                st.dataframe(df_display, use_container_width=True)

                # --- 📊 Bar Plot of Competition Scores ---
                st.write("### 📊 Competition Positioning Over Time")

                df_display['Highlight'] = 'Normal'
                df_display.loc[df_display['Date'] == best_row['Date'], 'Highlight'] = 'Best'
                df_display.loc[df_display['Date'] == worst_row['Date'], 'Highlight'] = 'Worst'

                color_map = {'Normal': 'rgba(66, 135, 245, 0.8)', 'Best': 'green', 'Worst': 'crimson'}

                fig_bar = px.bar(
                    df_display,
                    x='Date',
                    y='Competition (positioning)',
                    color='Highlight',
                    color_discrete_map=color_map,
                    text='Competition (positioning)',
                    title=f"Competition Results - {selected_name}",
                    labels={'Competition (positioning)': 'Position (Lower = Better)'}
                )

                fig_bar.update_traces(textposition='outside')
                fig_bar.update_yaxes(autorange='reversed')  # Lower = better
                fig_bar.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Competition Positioning",
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig_bar, use_container_width=True)

with tab6:
    st.write("### 🏆 Competition Predictor - S&C")

    # --- Filter by Area and Athlete ---
    filtered_area = st.selectbox("Select Area", ['S&C', 'Competition'], key="area_select_snc_final")
    available_names = sorted(df['Name'].dropna().unique())
    selected_name = st.selectbox("Select Athlete", available_names, key=f"competition_name_select_final_{filtered_area}")

    # --- Data Preparation ---
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Set'] = pd.to_numeric(df['Set'], errors='coerce')
    df['Rep'] = pd.to_numeric(df['Rep'], errors='coerce')
    df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')

    # --- Filter datasets ---
    df_snc = df[(df['Area'] == 'S&C') & (df['Name'] == selected_name)].copy()
    df_comp = df[(df['Area'] == 'Competition') & (df['Name'] == selected_name)].copy()

    # --- Prepare S&C weekly workload ---
    if not df_snc.empty:
        df_snc['Workload'] = df_snc['Set'] * df_snc['Rep'] * df_snc['Load (kg)']
        df_snc['Week from'] = (df_snc['Date'] - pd.to_timedelta(df_snc['Date'].dt.weekday, unit='d')).dt.normalize()
        df_weekly = (
            df_snc.dropna(subset=['Workload', 'Week from'])
            .groupby('Week from', as_index=False)['Workload']
            .mean()
        )
    else:
        df_weekly = pd.DataFrame(columns=['Week from', 'Workload'])

    # --- Prepare competition data ---
    if not df_comp.empty:
        df_comp['Competition (positioning)'] = pd.to_numeric(df_comp['Competition (positioning)'], errors='coerce')
        df_comp = df_comp.dropna(subset=['Competition (positioning)', 'Date']).sort_values('Date')
    else:
        df_comp = pd.DataFrame(columns=['Date', 'Competition (positioning)'])

    # --- Compute workload patterns between competitions ---
    training_patterns = []
    if not df_comp.empty and not df_weekly.empty:
        comp_dates = df_comp['Date'].tolist()
        comp_values = df_comp['Competition (positioning)'].tolist()

        for i in range(len(comp_dates)):
            start_date = comp_dates[i - 1] if i > 0 else None
            end_date = comp_dates[i]

            # Select S&C data between competitions (or before the first one)
            if start_date is not None:
                mask = (df_weekly['Week from'] > start_date) & (df_weekly['Week from'] <= end_date)
            else:
                mask = (df_weekly['Week from'] <= end_date)

            pre_period = df_weekly.loc[mask, ['Week from', 'Workload']].dropna().sort_values('Week from')

            if pre_period.empty:
                continue

            # Compute stats
            mean_workload = pre_period['Workload'].mean()
            workload_sd = pre_period['Workload'].std()
            workload_trend = pre_period['Workload'].diff().mean()

            if len(pre_period) > 1 and pre_period['Workload'].min() != 0:
                max_to_min_ratio = pre_period['Workload'].max() / pre_period['Workload'].min()
            else:
                max_to_min_ratio = np.nan

            last_week_workload = pre_period.iloc[-1]['Workload']
            pct_change_last2 = (
                (pre_period.iloc[-1]['Workload'] - pre_period.iloc[-2]['Workload'])
                / pre_period.iloc[-2]['Workload'] * 100
                if len(pre_period) > 1 else np.nan
            )

            # Append pattern
            pattern = {
                'Name': selected_name,
                'Competition_Date': end_date.date(),
                'Competition_Position': comp_values[i],
                'Mean_Workload': round(mean_workload, 2),
                'Workload_SD': round(workload_sd, 2) if pd.notna(workload_sd) else np.nan,
                'Last_Week_Workload': round(last_week_workload, 2),
                'Workload_Trend': round(workload_trend, 2) if pd.notna(workload_trend) else np.nan,
                'Max_to_Min_Ratio': round(max_to_min_ratio, 2) if pd.notna(max_to_min_ratio) else np.nan,
                'Pct_Change_Last2Weeks': round(pct_change_last2, 2) if pd.notna(pct_change_last2) else np.nan,
                'Weeks_counted': len(pre_period)
            }
            training_patterns.append(pattern)

    # --- Final Table ---
    pattern_df = pd.DataFrame(training_patterns)

    if not pattern_df.empty:
        # Remove duplicates & sort chronologically
        pattern_df = (
            pattern_df
            .drop_duplicates(subset=['Name', 'Competition_Date'])
            .sort_values('Competition_Date')
            .reset_index(drop=True)
        )

        st.write("### 📊 Computed Training Patterns (Chronological)")
        st.dataframe(
            pattern_df[
                ['Name', 'Competition_Date', 'Competition_Position',
                 'Mean_Workload', 'Workload_SD', 'Last_Week_Workload',
                 'Workload_Trend', 'Max_to_Min_Ratio',
                 'Pct_Change_Last2Weeks', 'Weeks_counted']
            ],
            use_container_width=True
        )
    else:
        st.info("No valid training pattern data found for the selected athlete.")
































































































