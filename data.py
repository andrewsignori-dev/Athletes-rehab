import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Load Excel data and clean column names & types."""
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()
    
    # Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.date
    
    # Numeric columns
    df['Load (kg)'] = pd.to_numeric(df['Load (kg)'], errors='coerce')
    if 'Tempo' in df.columns:
        df['Tempo'] = pd.to_numeric(df['Tempo'], errors='coerce')
    
    return df
