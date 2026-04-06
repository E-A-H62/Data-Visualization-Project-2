# Project 2 — Children Stories Visualizations

This project uses the **Children Stories Text Corpus** (downloaded via `kagglehub`) to create wordcloud-based visualizations.

## Two ways to view visualizations

- **`visualizations.py` (local / non-interactive script)**  
  Use this when you want to generate visualizations **locally** (e.g., in a Python session) and have plots pop up with Matplotlib (`plt.show()`).

- **`app.py` (interactive website)**  
  Use this when you want an **interactive web app** (built with Shiny for Python) where you can choose a story + text mode and click a button to generate a wordcloud in the browser.

## Quickstart

### Install dependencies

For Mac/Linux operating systems:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows operating systems:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Running Programs

### Create dataset
```bash
python dataset.py
```
Result should update into data/children_stories_dataset.csv

### Word Cloud Generation:
```bash
python visualizations.py
```
Word clouds displayed (unprocessed, stop word removal, lemmatization) on matplotlib window

### Zipf Demo: Run the interactive website
```bash
python app.py
```
Then open the printed local URL in your browser (the app binds to `0.0.0.0` and uses `PORT` when provided by a hosting platform).

