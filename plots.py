import plotly.express as px
import pandas as pd

def plot_load_trends(filtered_df, exercise_keywords):
    if filtered_df.empty:
        return None
    
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
        title='Average Load (kg) Over Time',
        template="plotly_white"
    )
    
    # Heatmap
    heatmap = px.density_heatmap(
        load_time_df,
        x='Month_Year',
        y='Exercise_Keyword',
        z='Load (kg)',
        color_continuous_scale='Blues',
        title='Heatmap of Average Load per Exercise Keyword Over Time',
        template="plotly_white"
    )
    
    return fig_bar, heatmap
