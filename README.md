# Wikipedia Fame Predictor

DS570 Final Project.

This project downloads daily Wikipedia pageviews for a list of famous
scientists and builds a small dashboard around them. It does two things:
it tries to predict tomorrow's pageviews, and it finds the days when a
scientist suddenly got a lot more attention than usual (usually because
of news, an award, or a death).

## How to Run it with Docker

You only need Docker installed.

```bash
docker build -t wiki-fame .
docker run -p 8501:8501 wiki-fame
```

Then open http://localhost:8501 in your browser.

The first time you open it, the app downloads about a year of data from
Wikipedia (this takes a minute or two), then trains the models and shows
the dashboard. The data is downloaded automatically, you do not need any
account or local files.

To run the tests:

```bash
docker run wiki-fame pytest -q
```

## The Data

The data comes from the Wikimedia Pageviews API, which is free and public
(no login needed). For each scientist we get the number of daily views on
their English Wikipedia page.

The list of scientists is in data/scientists.json. I picked around 74
people from different fields (physics, biology, chemistry, math, computer
science, astronomy, engineering). I included some recent Nobel Prize
winners on purpose, because their pages spike a lot around the time they
won, which makes the spike detection more interesting.

## The Dashboard

There are three tabs:

- Explore: pick a scientist and see their pageviews over the last year.
  There is also a bar chart of average views per field.
- Spikes: a table of the biggest recent spikes, and a chart that marks
  the spike days for one scientist.
- Prediction: compares the models on the last 30 days and shows the
  predicted vs real pageviews for a chosen scientist.

## How the Prediction Works

The number of pageviews is very skewed (some scientists get way more
views than others), so I predict log(views) instead of the raw number.

For each day I use a few simple features based on past days only:

- views yesterday (lag_1)
- views the same day last week (lag_7)
- average of the last 7 days (roll7)
- day of week and a weekend flag

I split the data by time: models are trained on everything except the
last 30 days, and tested on those last 30 days.

Models compared:
- Naive: predict yesterdays value
- Seasonal naive: predict the value from same day last week
- Linear Regression
- Random Forest

## How the Spike Detection Works

For each scientist I compute the rolling average and standard deviation
of the last 30 days, then a z-score for each day. If a day is 3 or more
standard deviations above normal, it is marked as a spike.

## Files

```
data/scientists.json   the list of scientists
src/data.py            downloads the pageviews from Wikipedia API
src/model.py           features, models, evaluation, spike detection
app/streamlit_app.py   the dashboard
tests/test_basic.py    offline tests
```

## Results

Model evaluation on the last 30 days (May 4 – June 2, 2026):

| Model | MAE (log) | MAPE (%) |
|---|---|---|
| Naive (yesterday) | 0.139 | 14.0 |
| Seasonal naive (last week) | 0.182 | 19.7 |
| Linear Regression | 0.122 | **12.0** |
| Random Forest | 0.131 | 13.3 |

Linear Regression wins on this holdout. The naive "predict yesterday"
baseline is surprisingly hard to beat (14.0%) — most days look a lot
like the day before. The Random Forest is close but not better here,
probably because the dataset is not large enough for it to show its
advantage.

Top spikes in the last 90 days (biggest z-scores):

| Date | Scientist | Views | Z-score |
|---|---|---|---|
| 2026-06-02 | Srinivasa Ramanujan | 12,906 | 10.6 |
| 2026-05-19 | Fei-Fei Li | 4,352 | 14.2 |
| 2026-04-30 | Norbert Wiener | 2,273 | 14.8 |
| 2026-05-15 | Ferenc Krausz | 1,060 | 11.5 |
| 2026-05-28 | Werner Heisenberg | 3,174 | 8.7 |

Most-viewed scientists on average (daily pageviews, last 365 days):

1. Albert Einstein — 15,546
2. Isaac Newton — 8,760
3. Marie Curie — 7,542
4. Srinivasa Ramanujan — 5,485
5. Thomas Edison — 4,757

## What it cannot do

The prediction only looks at past pageviews, so it cannot predict a spike
from an outside event like a Nobel announcement. That is expected, and it
is why the spike detection is a separate part of the project. It also only
uses English Wikipedia, and the API is about a day behind real time.
