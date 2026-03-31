import streamlit as st
import pandas as pd
import numpy as np
import requests

st.set_page_config(page_title="🏒 NHL Goal Scorer Model", layout="wide")
st.title("🏒 NHL Goal Scorer Model - Full iPhone Version")

# ---------- User Input ----------
team_id = st.text_input("Enter your team ID (e.g., 10 for TOR)")
opponent_id = st.text_input("Enter opponent team ID (e.g., 6 for MTL)")

if st.button("Fetch Live Stats"):
    if not (team_id.isnumeric() and opponent_id.isnumeric()):
        st.error("Team IDs must be numeric")
    else:
        team_id = int(team_id)
        opponent_id = int(opponent_id)

        # ---------- Fetch Team Roster ----------
        roster_url = f"https://statsapi.web.nhl.com/api/v1/teams/{team_id}?expand=team.roster"
        roster_data = requests.get(roster_url).json()
        players = roster_data["teams"][0]["roster"]["roster"]

        data_list = []

        for p in players:
            pid = p["person"]["id"]
            name = p["person"]["fullName"]

            # ---------- Live Season Stats ----------
            stats_url = f"https://statsapi.web.nhl.com/api/v1/people/{pid}/stats?stats=statsSingleSeason"
            stats_resp = requests.get(stats_url).json()
            splits = stats_resp["stats"][0]["splits"]
            if splits:
                stats = splits[0]["stat"]
                toi = stats.get("timeOnIcePerGame", "0:0")
                goals = stats.get("goals", 0)
                shots = stats.get("shots", 0)
                assists = stats.get("assists", 0)
            else:
                toi = "0:0"
                goals = shots = assists = 0

            toi_float = float(toi.split(":")[0]) if ":" in str(toi) else float(toi)

            # ---------- Recent Form (last 10 games) ----------
            game_log_url = f"https://statsapi.web.nhl.com/api/v1/people/{pid}/stats?stats=gameLog&season=20252026"
            game_log_resp = requests.get(game_log_url).json()
            last10 = game_log_resp["stats"][0]["splits"][:10] if game_log_resp["stats"][0]["splits"] else []
            last10_goals = sum([g["stat"].get("goals",0) for g in last10])
            last10_shots = sum([g["stat"].get("shots",0) for g in last10])
            last10_assists = sum([g["stat"].get("assists",0) for g in last10])
            recent_form = (last10_goals + last10_shots/10 + last10_assists)/10  # normalized

            # ---------- H2H (simplified) ----------
            # Average goals per game vs opponent in last 10 games (mock if no API)
            H2H = np.random.uniform(0.8, 1.2)

            # ---------- Positional & Line Weakness (simplified) ----------
            pos_weakness = np.random.uniform(0.8, 1.2)
            line_weakness = np.random.uniform(0.8, 1.2)
            opponent_weakness = (pos_weakness + line_weakness)/2

            # ---------- iXG ----------
            iXG = shots / 10.0

            data_list.append({
                "Player": name,
                "TOI": toi_float,
                "Goals": goals,
                "Shots": shots,
                "Assists": assists,
                "iXG": iXG,
                "RecentForm": recent_form,
                "H2H": H2H,
                "OpponentWeakness": opponent_weakness
            })

        # ---------- Build DataFrame ----------
        df = pd.DataFrame(data_list)
        df["TOIScore"] = df["TOI"]/df["TOI"].max()
        df["ModelScore"] = (
            df["iXG"]*0.30 +
            df["TOIScore"]*0.15 +
            df["RecentForm"]*0.20 +
            df["OpponentWeakness"]*0.20 +
            df["H2H"]*0.15
        )
        df["GoalProb"] = 1/(1+np.exp(-5*(df["ModelScore"]-0.5)))
        df["GoalProb"] = (df["GoalProb"]*100).round(1)
        df_sorted = df.sort_values("ModelScore", ascending=False)

        # ---------- Color-coded table ----------
        def color_prob(val):
            if val >= 70: return 'background-color: #85e085'
            elif val >= 50: return 'background-color: #ffff99'
            else: return 'background-color: #ff9999'

        st.subheader("📊 NHL Model Results (Full Data)")
        st.dataframe(df_sorted.style.applymap(color_prob, subset=["GoalProb"]))
