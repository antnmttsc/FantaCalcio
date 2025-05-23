import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from Utils.Load_data import (
    create_matchdays_df, 
    add_goals_info, 
    prepare_long_format_data, 
    build_scoreboard, 
    build_variance_data, 
    compute_expected_points
)

from Utils.EDA import (
    plot_team_points,
    plot_club_time_series,
    plot_points_distribution_violin,
    plot_goals_per_club,
    points_standing_per_points,
    plot_max_min_points,
    plot_longest_streaks,
    plot_robbed_points_subplots,
    plot_variance_by_wdl,
    plot_points_with_variance,
    plot_expected_points,
)

from Download_Guide import show_guide

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Dashboard", "Download Guide"])

if page == "Download Guide":
    show_guide()
    st.stop() 

if page == "Dashboard":
    st.title("⚽ Football Match Analysis Dashboard")

    # st.markdown("""
    # Upload your Excel file containing matchday data and the dashboard will do the rest!

    # Need help? Check out the **Download Guide** in the sidebar!
    # """)
    st.markdown("""
    Upload your Excel file containing matchday data and the dashboard will do the rest!

    :arrow_right: To learn how to download and prepare your file, go to the **Download Guide** tab in the sidebar.
    """)

    uploaded_file = st.file_uploader("Choose your Excel (.xlsx) file", type="xlsx")

    if uploaded_file:
        with st.spinner("Processing file and generating plots..."):
            all_matchdays = create_matchdays_df(uploaded_file)
            all_matchdays = add_goals_info(all_matchdays)

            # Section 1: Team Points Analysis
            st.header("Section 1: Team Points Analysis ⚽📊")
            st.markdown("""
            This chart displays the average points earned by each team when playing at home, away, and overall.  
            It provides insight into team performance across different match settings and highlights consistency.
            """)
            fig1 = plot_team_points(all_matchdays)  # Make sure this returns a figure
            st.pyplot(fig1)

            # Section 2: Points Distribution per Club
            club_df = prepare_long_format_data(all_matchdays)
            st.header("Section 2: Points Distribution By Club 🎯")
            st.markdown("""
            This chart shows how the points each team earns are spread across all their games in the season.\n
            The shape of each “violin” represents the range and frequency of points scored—wider sections mean the team scored those points more often.\n
            This helps you see not just the average points a team gets, but also how consistent or varied their performance is over time.\n
            For example, a narrow violin means the team scores similar points regularly, while a wider violin means their results vary a lot!
            """)
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            plot_points_distribution_violin(club_df, ax=ax2)
            st.pyplot(fig2)

            # Section 3: Team Points Over Matchdays
            st.header("Section 3: Points Trend Over Matchdays 📈")
            st.markdown("""
            "Ok fine the violin is nice but i don't care about the distribution, I just want to see how my team is doing over time!"\n
            Yeah sure, I get it, here's a line chart that shows how many points each team has earned over the matchdays.\n
            This visualization tracks points earned by each team over the course of the matchdays, revealing trends and performance fluctuations.
            """)
            fig3 = plot_club_time_series(all_matchdays)  # Make sure this returns a figure
            st.pyplot(fig3)

            # Section 4: Goals and Points Analysis
            st.header("Section 4: Goals and Points Analysis ⚽📊")
            st.markdown("""
            Let's take a look at the goals scored by each team
            """)
            fig4, ax4 = plt.subplots(figsize=(14,7))
            plot_goals_per_club(club_df, ax=ax4)
            st.pyplot(fig4)

            st.markdown("""
            Sure, you can already see this in the standings, but let's go a step further!
                        
            How many points does each team earn *per 100 goals scored*?
                        
            This gives you a fresh perspective on how efficient teams are at turning points into scoreboard points.
            """)
            scoreboard = build_scoreboard(club_df)
            fig5, ax5 = plt.subplots(figsize=(10,6))
            top_club, top_value, worst_club, worst_value = points_standing_per_points(scoreboard, ax=ax5)
            st.pyplot(fig5)
            st.markdown(f"""
            Did you get it? If not, don't worry now i will explain it to you!\n
            The chart shows how many points each team earns for every 100 goals they score.\n
            For example, if {top_club} scores 100 points in a single match, it will make a jump of {top_value} in the standings, easy right?
            (It's like {top_value/worst_value:.2f} times better than {worst_club}!)\n
            """)

            # Section 5: Max and Min Points per Club
            st.header("Section 5: Max and Min Points Per Club 📊")
            st.markdown("""
            Here we explore the highest and lowest points each club has scored in a match during the season.
            """)
            fig6, best_team, best_score, worst_team, worst_score = plot_max_min_points(club_df)
            st.pyplot(fig6)
            st.markdown(f"""
            Well done {best_team}! It is the best team with a score of {best_score} points in a single match!!\n
            But wait, {worst_team} is the worst team with a score of {worst_score} points... What a shame! How did you manage to do that?\n
            """)

            # Section 6: Longest Streaks per Club
            st.header("Section 6: Longest Streaks per Club 🔥")
            st.markdown("""
            Let's dive into the longest winning, losing, unbeaten, and winless streaks of each club throughout the season.
            """)
            fig7, ax7 = plt.subplots(2, 2, figsize=(14, 10))
            best_info = plot_longest_streaks(club_df, ax7)
            st.pyplot(fig7)
            st.markdown(f"""
            This visualization shows the longest streaks for each club during the season across four key categories:

            - **Top Left:** Longest winning streak. For example, **{best_info['win']['team']}** achieved the longest winning streak of **{best_info['win']['streak']}** games from Matchday **{best_info['win']['range'][0]}** to **{best_info['win']['range'][1]}**.
            - **Top Right:** Longest losing streak. For instance, **{best_info['lose']['team']}** faced the longest losing streak with **{best_info['lose']['streak']}** games between Matchday **{best_info['lose']['range'][0]}** and **{best_info['lose']['range'][1]}**.
            - **Bottom Left:** Longest unbeaten streak (no losses). **{best_info['not_lose']['team']}** remained unbeaten for **{best_info['not_lose']['streak']}** matches from Matchday **{best_info['not_lose']['range'][0]}** to **{best_info['not_lose']['range'][1]}**.
            - **Bottom Right:** Longest winless streak. **{best_info['not_win']['team']}** went through the longest winless run of **{best_info['not_win']['streak']}** games spanning Matchday **{best_info['not_win']['range'][0]}** to **{best_info['not_win']['range'][1]}**.
            """)

            # Section 7: Points Robbed from Each Club 🚨
            st.header("Section 7: Points Robbed from Each Club 🚨")
            st.markdown("""
            This section shows how many points each club has "had robbed" from them by their opponents throughout the season.

            - When a team loses, it’s like the opponent "robbed" 3 points.
            - When a team draws, the opponent "robbed" 2 points.
            - When a team wins, no points are robbed.

            Let's see who is your black sheep and for who your are the black sheep!
            """)
            fig8 = plot_robbed_points_subplots(all_matchdays)
            st.pyplot(fig8)

            # Section 8: Variance in Club Performance 🎢
            variance_by_club_df = build_variance_data(club_df, scoreboard)
            st.header("Section 8: Who's Got the Most Drama? 🎢")
            st.markdown("""
            Ever wonder which club is the rollercoaster of the league — full of ups, downs, and loop-the-loops?  
            Here, we dive into the variance of points each club racks up. The bigger the swings, the more *exciting* (or *stressful*) the ride!

            - Clubs with low standard deviation: steady as a rock 🪨 (boring but reliable).  
            - Clubs with high standard deviation: wild thrill-seekers 🎢 — expect surprises every matchday!  

            Let's find out who's the drama queen of the season and who's just chilling on the sidelines.
            """)

            fig9 = plot_variance_by_wdl(variance_by_club_df)
            st.pyplot(fig9)

            higer_var_club = variance_by_club_df.loc[variance_by_club_df['Variance'].idxmax(), 'Club']
            lower_var_club = variance_by_club_df.loc[variance_by_club_df['Variance'].idxmin(), 'Club']

            st.markdown(f"""
            Ready for a showdown? ⚔️  
            Pick two clubs below to see who’s been riding the rollercoaster of points more wildly this season.  

            Will it be a battle of the steady performers, or a clash of the unpredictable thrill-seekers?  
            Choose wisely and let the drama unfold!
            If you’re not sure who to pick, here’s a pro tip: try comparing **{higer_var_club}** — the club with the highest volatility —  
            against **{lower_var_club}** — the calm and steady one. Let the contrast speak for itself!
            """)

            clubs = club_df['Club'].unique()
            team_1 = st.selectbox("Pick the first club to compare:", clubs, index=0)
            team_2 = st.selectbox("Pick the second club to compare:", clubs, index=1 if len(clubs) > 1 else 0)

            if team_1 == team_2:
                st.warning("Please pick two different clubs to compare!")
            else:
                fig = plot_points_with_variance(club_df, scoreboard, clubs_to_plot=[team_1, team_2])
                st.pyplot(fig)


            # Section 9: Expected Points 🔮
            st.header("Section 9: Crystal Ball Time 🔮 — Expected Points Unveiled!")

            st.markdown("""
            Ever wish you had a magical crystal ball to see who’s *really* destined to shine this season? ✨ Well, we’re not sorcerers, but close enough!

            This section reveals each club’s **expected points** — the clever prediction of how many points they *should* rack up based on their performance so far.

            - Think of it as the universe’s way of saying, “Hey, here’s what’s likely to happen!”
            - Spoiler alert: some clubs might be outperforming their destiny, and others… well, let’s just say their stars are a bit dimmer. 🌟

            Ready to peek into the future? See who’s living up to the hype and who’s about to surprise us all. Your fantasy team’s fate might just depend on this!
            """)

            overall_expected_points = compute_expected_points(club_df)
            fig10, ax10 = plt.subplots(figsize=(14, 10))
            fig10, ax10 = plot_expected_points(overall_expected_points, ax=ax10)
            st.pyplot(fig10)

            st.markdown(f"""
            Here, we calculate the **expected points** for each club *at every matchday* — not just based on who they actually played, but imagining if they faced an average opponent from the league instead.  

            Think of it like this:  
            What if, for each matchday, your club wasn’t just facing that specific opponent, but a sort of “league average” challenger?  
            By doing this for all matchdays and then summing up those results, we get a predicted final scoreboard — a crystal ball glimpse at where each team *should* stand if things played out evenly.

            It’s a fair way to ask:  
            > *“If I were up against the league’s average strength every time, how many points would I likely end up with?”*

            ✨ **This is your league’s ‘what-if’ scoreboard — the ultimate reveal of how things might have shaken out if luck, drama, and randomness took a backseat.**  
            This above is the *true* power rankings — stripped of all the chaos and chance! ⚡️🔥
            """)

    else:
        st.info("Please upload an Excel file to start the analysis.")


# python -m streamlit run C:\Users\antom\OneDrive\Desktop\Fantacalcio\FantaCalcio\Streamlite_app.py

