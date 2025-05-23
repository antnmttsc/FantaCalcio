import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


#########################################
# Average Home and Away Points per Club #
#########################################

def plot_team_points(df, ax=None):

    # compute total points and match counts for home and away
    home_stats = df.groupby("Home")["Home_Points"].agg(["sum", "count"]).rename(columns={"sum": "Home_Total", "count": "Home_Matches"})
    away_stats = df.groupby("Away")["Away_Points"].agg(["sum", "count"]).rename(columns={"sum": "Away_Total", "count": "Away_Matches"})
    points_df = home_stats.join(away_stats)

    # compute avg
    points_df["Home Points"] = points_df["Home_Total"] / points_df["Home_Matches"]
    points_df["Away Points"] = points_df["Away_Total"] / points_df["Away_Matches"]

    # weighted Overall Points
    points_df["Overall Points"] = ((points_df["Home_Total"] + points_df["Away_Total"]) /
                                   (points_df["Home_Matches"] + points_df["Away_Matches"]))

    points_df = points_df.sort_values(by="Overall Points", ascending=False)

    x = np.arange(len(points_df))
    width = 0.25

    # Create figure and axes if not passed
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))
    else:
        fig = ax.figure  # get the figure from the passed axes

    bars1 = ax.bar(x - width, points_df['Home Points'], width, label='Home Points', color='darkblue')
    bars2 = ax.bar(x, points_df['Away Points'], width, label='Away Points', color='skyblue')
    bars3 = ax.bar(x + width, points_df['Overall Points'], width, label='Overall Points', color='gray')

    all_scores = np.concatenate([points_df['Home Points'].values,
                                 points_df['Away Points'].values,
                                 points_df['Overall Points'].values])
    ax.set_ylim((all_scores.min() - 5, all_scores.max() + 2))

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.05,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_title("Average Home, Away, and Overall Points for Each Team", fontsize=14)
    ax.set_xlabel("Team", fontsize=12)
    ax.set_ylabel("Average Points", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(points_df.index, rotation=45, ha='right')
    ax.legend()
    ax.grid(False)

    plt.tight_layout()

    return fig

######################
# Point Distribution #
######################

def plot_points_distribution_violin(club_df, ax=None):
    sorted_clubs = club_df.groupby('Club')['Points'].median().sort_values(ascending=False).index

    # Create figure and axes if none provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure

    sns.violinplot(
        x='Club',
        y='Points',
        data=club_df,
        inner="quart",
        order=sorted_clubs,
        hue='Club',
        palette='muted',
        legend=False,
        ax=ax
    )

    ax.set_xlabel("Club")
    ax.set_ylabel("Points")
    ax.set_title("Distribution of Points for Each Club")
    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()

    # Return figure if created here, otherwise None
    if ax is None:
        return fig

###############################
# Points Time Series per Club #
###############################

def plot_club_time_series(df, ax=None):

    club_names = df['Home'].unique()

    y_lim_min = min(df['Home_Points'].min(), df['Away_Points'].min()) - 1
    y_lim_max = max(df['Home_Points'].max(), df['Away_Points'].max()) + 1

    time_series = {club: [] for club in club_names}
    for _, row in df.iterrows():
        time_series[row["Home"].strip()].append((row["Matchday"], row["Home_Points"]))
        time_series[row["Away"].strip()].append((row["Matchday"], row["Away_Points"]))

    time_series_df = pd.DataFrame(columns=["Matchday"] + list(time_series.keys()))
    for club, scores in time_series.items():
        df_club = pd.DataFrame(scores, columns=["Matchday", club])
        if time_series_df.empty:
            time_series_df = df_club
        else:
            time_series_df = time_series_df.merge(df_club, on="Matchday", how="outer")

    time_series_df.set_index("Matchday", inplace=True)
    time_series_df.sort_index(inplace=True)

    overall_means = time_series_df.mean(axis=0)
    num_clubs = len(time_series)
    ncols = 3
    nrows = (num_clubs // ncols) + (num_clubs % ncols > 0)

    # If no ax provided, create subplots
    if ax is None:
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 4))
        ax = ax.flatten()
    else:
        fig = ax[0].figure if isinstance(ax, (list, np.ndarray)) else ax.figure
        if not isinstance(ax, (list, np.ndarray)):
            ax = [ax]  # ensure iterable

    colors = plt.colormaps.get_cmap('tab20').resampled(num_clubs)
    for i, club in enumerate(time_series.keys()):
        club_color = colors(i)
        ax[i].plot(time_series_df.index, time_series_df[club], marker="o", label=club, color=club_color)
        ax[i].axhline(y=overall_means[club], color=club_color, linestyle='--', label='Average')
        ax[i].set_title(club)
        ax[i].set_xlabel("Matchday")
        ax[i].set_ylabel("Points")
        ax[i].set_ylim(y_lim_min, y_lim_max)
        ax[i].grid(True, linewidth=0.5, alpha=0.3)
        ax[i].legend(loc="upper left")
        ax[i].set_xticks(time_series_df.index)
        ax[i].tick_params(axis='x', rotation=45, labelsize=8)

    # Remove any unused axes
    for j in range(num_clubs, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()

    return fig

####################
# GOALS AND POINTS #
####################

def plot_goals_per_club(club_df, ax=None):
    # Sum goals by club
    goals_per_club = club_df.groupby("Club")["Goals"].sum().sort_values(ascending=False)
    
    # Calculate y-axis limits
    y_min = max(0, goals_per_club.min() - 3)  # don't go below 0
    y_max = goals_per_club.max() + 2

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))

    # Plot vertical bar chart
    bars = ax.bar(goals_per_club.index, goals_per_club.values, color='skyblue')
    ax.set_ylabel('Total Goals')
    ax.set_xlabel('Club')
    ax.set_title('Total Goals Scored per Club')
    ax.set_ylim(y_min, y_max)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Add numbers above bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.3, f'{int(height)}', 
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if ax is None:
        return fig

def points_standing_per_points(score_board, ax=None):
    # Compute the percentage and sort descending
    wdl_percentage = round(100 * score_board["WDL"] / score_board["Points"], 2).sort_values(ascending=False)

    y_lim_min = max(wdl_percentage.min() - 1.5, 0)
    y_lim_max = wdl_percentage.max() + 0.3  # no need for min() here

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Create the plot
    bars = ax.bar(wdl_percentage.index, wdl_percentage, color='royalblue')

    # Add values on top of the bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.05, f'{height:.2f}',
                ha='center', va='bottom', fontsize=10)

    # Labels and title
    ax.set_title("Points in Standings per 100 Points from Matchdays", fontsize=14)
    ax.set_xlabel("Club", fontsize=12)
    ax.set_ylabel("Points in Standing", fontsize=12)
    ax.set_ylim(y_lim_min, y_lim_max)
    # ax.set_xticklabels(wdl_percentage.index, rotation=45, ha='right')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # Remove grid for cleaner look
    ax.grid(False)
    plt.tight_layout()

    # Get top club name and value
    top_club = wdl_percentage.index[0]
    top_value = wdl_percentage.iloc[0]

    worst_club = wdl_percentage.index[-1]
    worst_value = wdl_percentage.iloc[-1]

    if ax is None:
        return fig, top_club, top_value
    else:
        return top_club, top_value, worst_club, worst_value

##################
# Max-Min Points #
##################

def plot_max_min_points(club_df, ax=None):
    max_points_per_club = club_df.groupby("Club")["Points"].max()
    min_points_per_club = club_df.groupby("Club")["Points"].min()

    max_points_sorted = max_points_per_club.sort_values(ascending=False)
    min_points_sorted = min_points_per_club.sort_values(ascending=True)
    
    best_team = max_points_sorted.index[0]
    best_score = max_points_sorted.iloc[0]
    worst_team = min_points_sorted.index[0]
    worst_score = min_points_sorted.iloc[0]

    # Create figure and axes if not provided
    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    else:
        # If ax is provided, expect it to be an array of two axes
        axes = ax

    # Max points plot
    axes[0].bar(max_points_sorted.index, max_points_sorted.values, color='green')
    axes[0].set_title("Max Points for Each Club")
    axes[0].set_xlabel("Club")
    axes[0].set_ylabel("Points")
    axes[0].set_ylim(max_points_sorted.min() - 3, max_points_sorted.max() + 1)
    axes[0].grid(axis='y', alpha=0.3)

    axes[0].set_xticks(range(len(max_points_sorted.index)))
    axes[0].set_xticklabels(max_points_sorted.index, rotation=45, ha='right')

    for i, v in enumerate(max_points_sorted):
        axes[0].text(i, v + 0.05, str(v), ha='center', va='bottom', fontsize=10)

    # Min points plot
    axes[1].bar(min_points_sorted.index, min_points_sorted.values, color='red')
    axes[1].set_title("Min Points for Each Club")
    axes[1].set_xlabel("Club")
    axes[1].set_ylim(min_points_sorted.min() - 5, min_points_sorted.max() + 2)
    axes[1].grid(axis='y', alpha=0.3)

    axes[1].set_xticks(range(len(min_points_sorted.index)))
    axes[1].set_xticklabels(min_points_sorted.index, rotation=45, ha='right')

    for i, v in enumerate(min_points_sorted):
        axes[1].text(i, v + 0.05, str(v), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    if ax is None:
        return fig, best_team, best_score, worst_team, worst_score
    else:
        return fig, best_team, best_score, worst_team, worst_score

#######################
# Win and Lose Streak #
#######################

def plot_longest_streaks(club_df, ax=None):

    # Store streak values and their matchday ranges
    streaks = {
        "win": {}, "lose": {}, "not_lose": {}, "not_win": {}
    }
    ranges = {
        "win": {}, "lose": {}, "not_lose": {}, "not_win": {}
    }

    for team, data in club_df.groupby("Club"):
        data = data.sort_values(by="Matchday")  # Ensure chronological order

        # Initialize streak tracking
        win = lose = not_lose = not_win = 0
        win_start = lose_start = not_lose_start = not_win_start = None
        max_win = max_lose = max_not_lose = max_not_win = 0
        max_win_range = max_lose_range = max_not_lose_range = max_not_win_range = (None, None)

        for idx, row in data.iterrows():
            day = row["Matchday"]

            if row["WDL"] == 3:  # Win
                if win == 0: win_start = day
                win += 1
                if win > max_win:
                    max_win = win
                    max_win_range = (win_start, day)
                lose = 0
                not_win = 0
                if not_lose == 0: not_lose_start = day
                not_lose += 1
                if not_lose > max_not_lose:
                    max_not_lose = not_lose
                    max_not_lose_range = (not_lose_start, day)
            elif row["WDL"] == 0:  # Loss
                if lose == 0: lose_start = day
                lose += 1
                if lose > max_lose:
                    max_lose = lose
                    max_lose_range = (lose_start, day)
                win = 0
                not_lose = 0
                if not_win == 0: not_win_start = day
                not_win += 1
                if not_win > max_not_win:
                    max_not_win = not_win
                    max_not_win_range = (not_win_start, day)
            else:  # Draw
                win = lose = 0
                if not_lose == 0: not_lose_start = day
                not_lose += 1
                if not_lose > max_not_lose:
                    max_not_lose = not_lose
                    max_not_lose_range = (not_lose_start, day)
                if not_win == 0: not_win_start = day
                not_win += 1
                if not_win > max_not_win:
                    max_not_win = not_win
                    max_not_win_range = (not_win_start, day)

        # Save results
        for key, max_val, rng in zip(
            ["win", "lose", "not_lose", "not_win"],
            [max_win, max_lose, max_not_lose, max_not_win],
            [max_win_range, max_lose_range, max_not_lose_range, max_not_win_range]
        ):
            streaks[key][team] = max_val
            ranges[key][team] = rng

    # Convert to sorted DataFrames with ranges
    dfs = {}
    best_streaks_info = {}

    for key in ["win", "lose", "not_lose", "not_win"]:
        df_temp = pd.DataFrame([
            {"Club": club, "Streak": streaks[key][club], "Range": ranges[key][club]}
            for club in streaks[key]
        ])
        df_sorted = df_temp.sort_values(by="Streak", ascending=False)
        dfs[key] = df_sorted

        # Get best team info as tuple (start, end)
        if not df_sorted.empty:
            best_row = df_sorted.iloc[0]
            best_streaks_info[key] = {
                "team": best_row["Club"],
                "streak": best_row["Streak"],
                "range": best_row["Range"]  # this is already a tuple (start, end)
            }
        else:
            best_streaks_info[key] = {"team": None, "streak": 0, "range": (None, None)}

    titles = {
        "win": "Longest Win Streaks",
        "lose": "Longest Lose Streaks",
        "not_lose": "Longest Not Lose Streaks",
        "not_win": "Longest Not Win Streaks"
    }

    colors = {
        "win": "green",
        "lose": "red",
        "not_lose": "blue",
        "not_win": "orange"
    }

    # Flatten axes in case a 2x2 numpy array is passed
    axes_flat = ax.flatten() if hasattr(ax, "flatten") else [ax]

    for i, key in enumerate(["win", "lose", "not_lose", "not_win"]):
        df_plot = dfs[key]
        current_ax = axes_flat[i]
        current_ax.clear()
        bars = current_ax.barh(df_plot["Club"], df_plot["Streak"], color=colors[key])
        current_ax.set_title(titles[key])
        current_ax.set_xlabel("Games")
        current_ax.set_ylabel("Club")
        current_ax.set_xticks(range(0, int(df_plot["Streak"].max()) + 1))
        current_ax.grid(True, linewidth=0.5, alpha=0.3)

        # Annotate bars with the Matchday range
        for bar, (start, end) in zip(bars, df_plot["Range"]):
            width = bar.get_width()
            if width > 0:
                current_ax.text(width - 0.5, bar.get_y() + bar.get_height()/2,
                                f"{start} → {end}", va='center', ha='right',
                                color='white', fontsize=8, fontweight='bold')

    plt.tight_layout()
    return best_streaks_info

#################
# Points robbed #
#################

def compute_points_robbed_from_each_club(df):

  clubs = df['Home'].unique()
  robbed_dict = {club: {} for club in clubs}

  for club in clubs:
    matches = df[(df['Home'] == club) | (df['Away'] == club)]

    for _, row in matches.iterrows():
      if row['Home'] == club:
        club_goals = row['Home_Goals']
        opp_goals = row['Away_Goals']
        opponent = row['Away']
      else:
        club_goals = row['Away_Goals']
        opp_goals = row['Home_Goals']
        opponent = row['Home']

      # Points robbed logic
      if club_goals > opp_goals:
        robbed_points = 0
      elif club_goals < opp_goals:
        robbed_points = 3
      else:
        robbed_points = 2

      if robbed_points > 0:
        if opponent in robbed_dict[club]:
          robbed_dict[club][opponent] += robbed_points
        else:
          robbed_dict[club][opponent] = robbed_points

  return robbed_dict

def plot_robbed_points_subplots(df):
    robbed_data = compute_points_robbed_from_each_club(df)
    teams = list(robbed_data.keys())
    num_teams = len(teams)
    
    cols = 3
    rows = (num_teams + cols - 1) // cols  # calculate number of rows needed
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes_flat = axes.flatten() if rows > 1 else [axes]  # flatten axes array if multiple rows
    
    for i, team in enumerate(teams):
        ax = axes_flat[i]
        opponents_points = list(robbed_data[team].items())
        opponents_points.sort(key=lambda x: x[1], reverse=True)
        
        opponents = [op for op, pts in opponents_points]
        points = [pts for op, pts in opponents_points]

        ax.clear()
        ax.bar(range(len(opponents)), points, color='skyblue')
        ax.set_title(f'Points Robbed from {team}')
        ax.set_ylabel('Points Robbed')

        ax.set_xticks(range(len(opponents)))
        ax.set_xticklabels(opponents, rotation=45, ha='right')

        y_lim_min = min(points) - 1 if points and min(points) > 2 else 0
        y_lim_max = max(points) + 1 if points else 1
        ax.set_ylim(y_lim_min, y_lim_max)
        ax.grid(axis='y', alpha=0.3)
    
    # Hide unused axes if any (when number of teams < rows*cols)
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    
    plt.tight_layout()
    return fig

#########################
# Variance Distribution #
#########################

def plot_variance_by_wdl(variance_by_club_df):
    # Sort by WDL descending
    sorted_df = variance_by_club_df.sort_values(by='WDL', ascending=False)

    # Normalize WDL for color mapping
    norm = mcolors.Normalize(vmin=sorted_df['WDL'].min(), vmax=sorted_df['WDL'].max())
    cmap = mcolors.LinearSegmentedColormap.from_list("blue_lightblue", ["blue", "lightblue"])

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_df['Club'],
                  sorted_df['Variance'],
                  color=cmap(norm(sorted_df['WDL'])))

    # Add WDL values on top of bars
    for bar, wdl in zip(bars, sorted_df['WDL']):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.05, f'{wdl}',
                ha='center', va='bottom', fontsize=10)

    # Set y-limits with a margin
    ymin = sorted_df['Variance'].min() - 0.5
    ymax = sorted_df['Variance'].max() + 0.5
    ax.set_ylim(ymin, ymax)

    ax.set_title('Standard Deviation Sorted by Points in Standings', fontsize=14)
    ax.set_xlabel('Club', fontsize=12)
    ax.set_ylabel('Standard Deviation', fontsize=12)
    # ax.set_xticklabels(sorted_df['Club'], rotation=45, ha='right')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', linewidth=0.3)
    fig.tight_layout()

    return fig

def plot_points_with_variance(club_df, score_board, clubs_to_plot=None):
    
    # Default clubs: first and last from score_board index if not specified
    if clubs_to_plot is None:
        clubs_to_plot = score_board.index[[0, -1]].tolist()
    
    # Prepare colors cycling through matplotlib defaults
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    club_colors = {club: default_colors[i % len(default_colors)] for i, club in enumerate(clubs_to_plot)}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for club in clubs_to_plot:
        club_data = club_df[club_df["Club"] == club].sort_values(by="Matchday")
        
        std_dev_points = np.sqrt(club_data["Points"].var())
        
        ax.plot(club_data["Matchday"], club_data["Points"],
                label=club, marker='o', color=club_colors[club])
        
        ax.fill_between(club_data["Matchday"],
                        club_data["Points"] - std_dev_points,
                        club_data["Points"] + std_dev_points,
                        color=club_colors[club], alpha=0.3)
    
    ax.set_title("Points Time Series with Standard Deviation", fontsize=14)
    ax.set_xlabel("Matchday", fontsize=12)
    ax.set_ylabel("Points", fontsize=12)
    ax.legend(title="Club")
    ax.grid(True, linewidth=0.3)
    
    # Rotate x-axis labels for readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    fig.tight_layout()
    
    return fig

###################
# Win Probability #
###################

def plot_expected_points(expected_points_dict, ax=None):
    sorted_items = sorted(expected_points_dict.items(), key=lambda x: x[1], reverse=True)
    clubs = [item[0] for item in sorted_items]
    points = [item[1] for item in sorted_items]

    cmap = plt.get_cmap("tab20")
    club_colors = {club: cmap(i % cmap.N) for i, club in enumerate(clubs)}
    bar_colors = [club_colors[club] for club in clubs]

    # If no axes passed, create one
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure  # get figure from axes

    bars = ax.barh(clubs, points, color=bar_colors)
    
    ax.set_xlabel('Expected Points', fontsize=12)
    ax.set_title('Overall Expected Points by Club', fontsize=14)
    ax.invert_yaxis()
    ax.set_xlim(int(min(points) - 5), int(max(points) + 1.6))
    ax.grid(axis='x', linewidth=0.3)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{int(width)}', va='center', fontsize=10)

    fig.tight_layout()
    return fig, ax















