# Wikipedia Fame Predictor

DS 570 final project.

This project downloads daily Wikipedia pageviews for a list of famous
scientists and builds a small dashboard around them. It does two things:
it tries to predict tomorrow's pageviews, and it finds the days when a
scientist suddenly got a lot more attention than usual (usually because
of news, an award, or a death).

## How to run it with Docker

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

## The data

The data comes from the Wikimedia Pageviews API, which is free and public
(no login needed). For each scientist we get the number of daily views on
their English Wikipedia page.

The list of scientists is in data/scientists.json. I picked around 74
people from different fields (physics, biology, chemistry, math, computer
science, astronomy, engineering). I included some recent Nobel Prize
winners on purpose, because their pages spike a lot around the time they
won, which makes the spike detection more interesting.

## The dashboard

There are three tabs:

- Explore: pick a scientist and see their pageviews over the last year.
  There is also a bar chart of average views per field.
- Spikes: a table of the biggest recent spikes, and a chart that marks
  the spike days for one scientist.
- Prediction: compares the models on the last 30 days and shows the
  predicted vs real pageviews for a chosen scientist.

## How the prediction works

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

## How the spike detection works

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

## What it cannot do

The prediction only looks at past pageviews, so it cannot predict a spike
from an outside event like a Nobel announcement. That is expected, and it
is why the spike detection is a separate part of the project. It also only
uses English Wikipedia, and the API is about a day behind real time.
