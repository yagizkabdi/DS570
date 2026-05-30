# Wikipedia Fame Predictor

DS 570 final project.

This project downloads daily Wikipedia pageviews for a list of famous
scientists and builds a small dashboard around them. It tries to predict
tomorrow's pageviews, and finds the days when a scientist suddenly got a
lot more attention than usual.

## How to run

You only need Docker.

```bash
docker build -t wiki-fame .
docker run -p 8501:8501 wiki-fame
```

Open http://localhost:8501. The first run downloads a year of data from
Wikipedia (~1-2 min), trains the models, then shows the dashboard.
No accounts or local files needed.

Run the tests:

```bash
docker run wiki-fame pytest -q
```
