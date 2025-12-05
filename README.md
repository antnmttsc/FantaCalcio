# ⚽ FantaCalcio Matchday Analyzer

A **Streamlit web app** to analyze fantasy football matchday data from an Excel file.  
Upload your `.xlsx` file and get automatic analysis including team points, goals, performance trends, streaks, variance, and expected points.
The app works only with the files from [this app](https://leghe.fantacalcio.it/).
---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/antnmttsc/FantaCalcio.git
cd FantaCalcio
```

### 2. Install dependencies
I recommend using a virtual environment.

```bash
pip install -r requirements.txt
```
### 3. Run the Streamlit app

```
streamlit run Streamlite_app.py
```

## Required Excel Format
Must be a valid .xlsx file

Data format and structure are explained in the Download Guide tab inside the app

## Key Features
- Home, away, and total points per team

- Violin plot showing points distribution by club

- Team points evolution over matchdays (line chart)

- Goals per club and efficiency (points per 100 goals)

- Maximum and minimum match scores

- Longest win/loss/unbeaten/winless streaks

- "Robbed points" by opponents (who stole your victories?)

- Performance variance across teams

- Expected Points (based on average-league simulations)

