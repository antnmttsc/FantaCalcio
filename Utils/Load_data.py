import pandas as pd
import numpy as np

# CREATE MATCHDAY DATAFRAME
def load_data(file_obj):
    # file_obj can be a path string or a file-like object from Streamlit uploader
    df = pd.read_excel(file_obj, skiprows=2)
    df.columns = list(range(0, 11))
    return df

def process_matchdays(df, cols):

    df_sub = df[cols].copy()  # pick columns for even or odd matchdays
    df_sub.columns = ['0', '1', '2', '3']

    # identify rows containing matchday headers in column '0'
    df_sub['Matchday'] = df_sub['0'].where(df_sub['0'].str.contains('Giornata', na=False))

    # forward fill the Matchday column
    df_sub['Matchday'] = df_sub['Matchday'].ffill()

    # extract numeric matchday number
    df_sub['Matchday'] = df_sub['Matchday'].str.extract(r'(\d+)').astype(int)

    # drop rows where any cell contains 'Giornata' s.t. we only keep the match data
    df_sub = df_sub[~df_sub.apply(lambda row: row.astype(str).str.contains('Giornata').any(), axis=1)].reset_index(drop=True)

    return df_sub

def create_matchdays_df(file_obj):
  
  df_raw = load_data(file_obj)

  odd_matchdays = process_matchdays(df_raw, [0,1,2,3])
  even_matchdays = process_matchdays(df_raw, [6,7,8,9])

  all_matchdays = pd.concat([odd_matchdays, even_matchdays], ignore_index=True)
  all_matchdays = all_matchdays.sort_values(by='Matchday').reset_index(drop=True)

  # rename columns
  all_matchdays.columns = ['Home', 'Home_Points', 'Away_Points', 'Away', 'Matchday']

  # remove null matchdays
  zero_points_matchdays = all_matchdays.groupby('Matchday').filter(
      lambda g: (g['Home_Points'] == 0).all() and (g['Away_Points'] == 0).all()
  )
  matchdays_with_all_zeros = zero_points_matchdays['Matchday'].unique()
  all_matchdays = all_matchdays[~all_matchdays['Matchday'].isin(matchdays_with_all_zeros)]

  return all_matchdays

# ADD GOALS INFO
def points_to_goals(points, min_thrs=66, points_per_goals=5):
    if points < min_thrs:
        return 0
    return (points - min_thrs) // points_per_goals + 1

def compute_points(row):
    if row['Home_Goals'] > row['Away_Goals']:
        return 3, 0
    elif row['Home_Goals'] < row['Away_Goals']:
        return 0, 3
    else:
        return pd.Series([1, 1])
        
def add_goals_info(df):

    df['Home_Goals'] = df['Home_Points'].apply(points_to_goals)
    df['Away_Goals'] = df['Away_Points'].apply(points_to_goals)

    df[["WDL_Home", "WDL_Away"]] = df.apply(compute_points, axis=1)
    return df

# PREPARE LONG FORMAT DATA
def prepare_long_format_data(df):
    club_data = []
    for _, row in df.iterrows():
        club_data.append({"Club": row["Home"], "Points": row["Home_Points"], "Home_or_Away": "Home", "Goals": row["Home_Goals"], "WDL": row["WDL_Home"], "Matchday": row["Matchday"]})
        club_data.append({"Club": row["Away"], "Points": row["Away_Points"], "Home_or_Away": "Away", "Goals": row["Away_Goals"], "WDL": row["WDL_Away"], "Matchday": row["Matchday"]})

    club_df = pd.DataFrame(club_data)
    return club_df

# BUILD SCOREBOARD
def build_scoreboard(club_df):
    total_points = club_df.groupby("Club")["Points"].sum()
    total_goals = club_df.groupby("Club")["Goals"].sum()
    total_wdl = club_df.groupby("Club")["WDL"].sum()

    score_board = pd.merge(total_points, total_goals, on="Club")
    score_board = pd.merge(score_board, total_wdl, on="Club")

    score_board = score_board.sort_values(by="WDL", ascending=False)

    return score_board

# VARAINCE DATA
def build_variance_data(club_df, score_board):
    variance_by_club = np.sqrt(club_df.groupby("Club")["Points"].var())
    variance_by_club = variance_by_club.sort_values(ascending=False)
    variance_by_club.name = "Variance"
    # Convert the Series into a DataFrame and reset the index (this will convert the index into a column)
    variance_by_club_df = variance_by_club.reset_index()

    # Rename the index column to 'Club'
    variance_by_club_df.rename(columns={'index': 'Club'}, inplace=True)

    # Merge with the score_board DataFrame on the "Club" column
    variance_by_club_df = pd.merge(variance_by_club_df, score_board, on="Club")

    return variance_by_club_df

# EXPECTED POINTS
def compute_expected_points(club_df):
    
  clubs_dict = {club: [] for club in club_df['Club'].unique()}

  for match_d in club_df['Matchday'].unique():
    match_d_df = club_df[club_df['Matchday'] == match_d]

    for club in match_d_df['Club'].unique():
      club_goal = match_d_df[match_d_df['Club'] == club]['Goals'].values[0]

      numb_win, numb_lose, numb_drawn = 0, 0, 0

      for other_club in match_d_df['Club'].unique():
        if club != other_club:
          other_club_goal = match_d_df[match_d_df['Club'] == other_club]['Goals'].values[0]

          if club_goal > other_club_goal:
            numb_win += 1
          elif club_goal < other_club_goal:
            numb_lose += 1
          else:
            numb_drawn += 1

      total = numb_win + numb_lose + numb_drawn
      
      if total > 0:
        win_prob = numb_win / total
        drawn_prob = numb_drawn / total

        expected_points = 3 * win_prob + 1 * drawn_prob
        clubs_dict[club].append(expected_points)

  overall_expected_points = {club: round(sum(points)) for club, points in clubs_dict.items()}
  return overall_expected_points
