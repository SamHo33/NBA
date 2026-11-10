# Does draft position predict NBA success?

Every June, NBA teams pick new players in the draft. The order of selection (the pick number) reflects how good teams expect each player to be.
This project asks: **using only what a team knows on draft night, how well can I predict a player's early career, and which picks beat or missed expectations?**

## Key results

- **Draft position matters, but it isn't destiny.** 72% of top-5 picks become useful players, compared with 8% of second-round picks, yet every pick range produces both busts and stars.
- **Draft-night information gives a clear but limited edge.** The best model explains about 23% of early-career success (R² 0.23) and ranks a real hit above a non-hit 78% of the time (AUC 0.78). The pick number does almost all the work.
- **Steals come from anywhere; big busts only from the top.** Nikola Jokić (pick 41) and Isaiah Thomas (pick 60) far beat their draft slots, while the biggest busts were all top-5 picks.

## Data

- **Source:** [NBA Player Statistics (1947–Present)](https://www.kaggle.com/datasets/sumitrodatta/nba-aba-baa-stats) on Kaggle, by sumitrodatta (downloaded November 2026; the dataset is updated each season)
- **Files used:**
  - `Draft Pick History.csv`: one row per draft pick
  - `Player Career Info.csv`: birth date, height, weight, position
  - `Advanced.csv`: season-by-season stats, including win shares
- **Scope:** NBA drafts from 1989 to 2018 (**1,742 picks**)

## Project structure

| Notebook | What it does | Output |
|---|---|---|
| `notebooks/01_build_dataset.ipynb` | Builds one clean table with one row per draft pick | `data/processed/draft_players.csv` |
| `notebooks/02_define_targets.ipynb` | Defines what "success" means and adds the `hit` column | `data/processed/draft_players_targets.csv` |
| `notebooks/03_explore.ipynb` | Explores how success changes with pick number, era, college background and age | charts only |
| `notebooks/04_models.ipynb` | Builds and compares models that use only draft-night information | results only |
| `notebooks/05_steals_busts.ipynb` | Finds the players and franchises that most beat or missed expectations | results only |

Each notebook reads the previous step's output and writes its own file, so any step can be rerun without breaking the others.

## Step 1: Building the dataset

WHAT: Joined tables on a unique **player ID** instead of player name. 
WHY: Different players can share a name, so joining on names can mix them up. 

WHAT: **Kept the 240 picks who never played**, giving them 0 win shares. 
WHY: Removing them would make late picks look better than they really were (**survivorship bias**). 

WHAT: Fixed **traded players being counted twice**. 
WHY: For a traded season, the data has a season-total row *and* one row per team. Adding them all double counted that season, so I keep only the total row (31,560 → 25,760 season rows). 

WHAT: Stopped at the **2018 draft**. 
WHY: Every player gets at least 4 full seasons, so recent picks are not unfairly judged. 

WHAT: Left age, height and position **blank** for picks who never played. 
WHY: That information is only in the data for players who reached the NBA. A blank would mean "never played" 100% of the time, so a model using it would be cheating (**target leakage**). The main model will only use facts known for every pick.

## Step 2: Defining success

A **win share** is an estimate of how many of a team's wins one player was responsible for. Roughly, 2–3 per season is a useful rotation player, and 10+ per season is an All-Star level player.

I use two targets:

- **`ws_first4`**: win shares in the 4 seasons right after the draft. This is roughly the rookie contract, the years the drafting team controls. Seasons a player missed (injured, on the bench, or playing overseas) count as 0, because the team still paid that cost.

- **`hit`**: True if `ws_first4` is at least **10** (about 2.5 per season, a solid rotation player). **419 of 1,742 picks (24%)** are hits.

## Step 3: Exploring the data

**How success changes with the pick**

| Pick range | What it is | Hit rate |
|---|---|---|
| 1–5 | the very top picks | 72% |
| 6–14 | the rest of the lottery | 44% |
| 15–30 | the rest of the first round | 26% |
| 31–60 | the second round | 8% |

- The expected value of a pick drops steeply from pick 1 to about pick 15, flattens around picks 15–25, and falls close to zero once the second round starts (about pick 31).
- Every range has busts and stars, and the ranges overlap: a good late-first-round pick often matches a weak top-5 pick.

**Other draft-night facts**
- **Era:** top-5 picks hit less often in 2009–18 (56%) than in 1989–98 (82%). With only 50 top-5 picks per era, this is a pattern worth noting, not a proven trend.
- **College background:** players with no college listed (mostly international) hit less often at every pick range. Part of this may be because some stay overseas after the draft, which counts as 0 in the 4-season window.
- **Age:** younger draftees do better overall (correlation −0.21), but within the same pick range the effect mostly disappears. Teams already draft younger players earlier, so age mostly repeats what the pick number shows (**confounding**).

## Step 4: Building models

I trained on drafts from 1989–2010 and tested on 2011–2018, so the models are graded on drafts they never saw, the way a team would use them. Features are limited to what is known for every pick on draft night: pick number, log of the pick number, college background and draft year.

**Predicting a hit (yes/no)**
- Only 22% of test picks are hits, so always guessing "not a hit" is 78% accurate while finding no hits at all. I used **AUC** instead, which measures how well the model ranks hits above non-hits.
- Logistic regression on the log of the pick reaches **AUC 0.77**; adding other features or gradient boosting only raises it to 0.78.
- **Sensitivity check:** with hit thresholds of 8, 10 and 15 win shares, AUC is 0.79, 0.78 and 0.74. Stars are harder to predict than useful players, but the conclusion doesn't depend on the threshold.

**What this means:** the pick number carries almost all the predictive information. College background helps slightly, draft year makes the linear model worse (it extends a trend into years it hasn't seen), and a more flexible model is only slightly better than a simple one. About three quarters of early-career success can't be predicted from draft-night data.

## Step 5: Steals and busts

For every pick, I compared actual win shares with the expected value for that draft slot (from a regression on the log of the pick number: about 23 for pick 1, under 1 for pick 60). The difference is the **surplus**.

| | Players |
|---|---|
| **Biggest steals** | Chris Paul (pick 4), Shawn Marion (9), LeBron James (1), Paul Pierce (10), Dirk Nowitzki (9) |
| **Best second-round picks** | Nikola Jokić (41), Paul Millsap (47), Isaiah Thomas (60), Draymond Green (35), Marc Gasol (48) |
| **Biggest busts** | Michael Olowokandi (1), Anthony Bennett (1), Markelle Fultz (1), Adam Morrison (3), Jay Williams (2) |

- Big busts only come from the top of the draft, because only high picks carry high expectations to miss. Some busts were caused by injuries rather than bad scouting.
- **Franchises:** from 1989 to 2018, Detroit, Houston and Golden State drafted most above expectations, and the Clippers furthest below. Most teams were close to expectations. This describes how each franchise's picks turned out; it doesn't prove any current front office drafts better, and the credit goes to the team that made the pick even if the player was traded right away.


## Changes from the original version

The original version of this project (a university assignment) predicted **career points** from **career minutes**, PER and win shares, and reached R² = 0.94 (0.976 with polynomial features). The score looked strong, but career minutes are only known *after* a career, and players who play more minutes naturally score more points. The model couldn't help a team on draft night.

This version:
- predicts success using only information available **on draft night**
- fixes the double counting of traded players and the name-based join
- keeps players who never played, to avoid survivorship bias
- tests models on later drafts they never saw, instead of a random split

The new best R² is 0.23 instead of 0.94. It is lower, but it is the honest answer to the question a team actually faces on draft night.

## How to run

1. Download the Kaggle dataset above and put its CSV files in `data/archive/`.
2. Install Python with `pandas`, `numpy`, `matplotlib`, `scikit-learn` and `jupyter` (for example, through Anaconda).
3. Run the notebooks in order, from `01_build_dataset` to `05_steals_busts`.
