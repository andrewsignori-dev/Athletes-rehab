import pandas as pd

def week_start_date(week: int, year: int) -> pd.Timestamp:
    return pd.to_datetime(f'{year}-W{int(week)}-1', format='%G-W%V-%u')

def week_end_date(week: int, year: int) -> pd.Timestamp:
    return pd.to_datetime(f'{year}-W{int(week)}-7', format='%G-W%V-%u')

def filter_by_keywords(df: pd.DataFrame, column: str, keywords: list) -> pd.DataFrame:
    """Filter rows where any keyword appears in the specified column."""
    if not keywords or column not in df.columns:
        return df
    pattern = '|'.join([kw.strip() for kw in keywords])
    return df[df[column].str.contains(pattern, case=False, na=False)]
