import pandas as pd

def apply_filters(df: pd.DataFrame, selected_areas=[], selected_names=[], selected_years=[],
                  selected_months=[], load_filter='All', load_range=None, families=[]):
    filtered = df.copy()
    
    if selected_areas:
        filtered = filtered[filtered['Area'].isin(selected_areas)]
    if selected_names:
        filtered = filtered[filtered['Name'].isin(selected_names)]
    if selected_years:
        filtered = filtered[filtered['Date'].apply(lambda x: x.year).isin(selected_years)]
    if selected_months:
        filtered = filtered[filtered['Date'].apply(lambda x: x.month).isin(selected_months)]
    if families and 'Family' in df.columns:
        filtered = filtered[filtered['Family'].isin(families)]
    
    # Load filter
    if load_filter == "With Load":
        filtered = filtered[filtered['Load (kg)'].notna()]
        if load_range:
            filtered = filtered[filtered['Load (kg)'].between(load_range[0], load_range[1])]
    elif load_filter == "Without Load":
        filtered = filtered[filtered['Load (kg)'].isna()]
    
    return filtered
