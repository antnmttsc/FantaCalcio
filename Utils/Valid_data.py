import pandas as pd
import streamlit as st

# VALIDATE MATCHDAY DATAFRAME
def validate_matchdays_df(df):
    try:
        # Ensure DataFrame has the expected structure
        expected_columns = ['Home', 'Home_Points', 'Away_Points', 'Away', 'Matchday']
        if list(df.columns) != expected_columns:
            raise ValueError(f"Expected columns: {expected_columns}, but got: {list(df.columns)}")

        # 1. Home and Away must be strings with < 16 unique values, and the same team set
        if not (df['Home'].dtype == 'object' and df['Away'].dtype == 'object'):
            raise TypeError("Home and Away team names must be strings.")

        home_teams = set(df['Home'].unique())
        away_teams = set(df['Away'].unique())

        if len(home_teams) > 16 or len(away_teams) > 16:
            raise ValueError("Too many unique teams. Maximum allowed is 16.")

        if home_teams != away_teams:
            raise ValueError("Mismatch between Home and Away team names. They must be identical sets.")

        # 2. Points must be numeric, non-negative, <= 300
        for col in ['Home_Points', 'Away_Points']:
            # Try to convert to numeric, coerce errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            if df[col].isnull().any():
                raise TypeError(f"{col} contains non-numeric values that couldn't be converted.")

            if (df[col] < 0).any() or (df[col] > 300).any():
                raise ValueError(f"{col} values must be between 0 and 300.")


        # 3. Matchday must be a continuous sequence (e.g., 1 to 34)
        matchdays = sorted(df['Matchday'].unique())
        if matchdays != list(range(matchdays[0], matchdays[-1] + 1)):
            raise ValueError("Matchday values must be a continuous sequence (e.g., 1-34, 5-32).")

        return df  # All checks passed

    except Exception as e:
        st.error(f"❌ Validation Error: {e}")
        return None

