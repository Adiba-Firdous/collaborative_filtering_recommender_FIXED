# Collaborative Filtering Recommender (Surprise)

A complete educational movie recommender built with Python and the Surprise
library. The project demonstrates a user-item rating matrix, cosine similarity,
top-N recommendations, Precision@K / Recall@K, and recommendations for a new user.

## Features

- User-based KNN collaborative filtering for existing users.
- Item-based cosine similarity for new-user recommendations.
- Excludes items the user has already rated.
- Similar-user lookup.
- Precision@K and Recall@K evaluation on a held-out test split.
- Sample movie and ratings CSV files included.
- Tests for core functionality.

## Folder structure

```text
collaborative_filtering_recommender/
├── data/
│   ├── movies.csv
│   └── ratings.csv
├── src/
│   ├── __init__.py
│   ├── recommender.py
│   └── evaluate.py
├── tests/
│   └── test_recommender.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Requirements

Use Python 3.10 or 3.11. Surprise contains compiled extensions; Python 3.11
is recommended for a straightforward installation.

### Windows setup

Open VS Code terminal in this project directory:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run

```powershell
python main.py
```

### Run tests

```powershell
pytest -q
```

## How the algorithm works

1. The ratings CSV contains user IDs, item IDs, and ratings from 1 to 5.
2. Surprise creates a trainset from the user-item-rating records.
3. The user-based KNN model finds users with similar rating patterns.
4. The model predicts scores for items the selected user has not rated.
5. The item-based KNN model compares items and estimates scores for a new
   user's unseen items from the few ratings they provide.
6. Evaluation uses held-out ratings. A rating of 4 or 5 is considered relevant
   for Precision@K and Recall@K.

## Data format

`data/ratings.csv`:

```csv
user_id,item_id,rating
1,1,5
1,2,4
```

`data/movies.csv`:

```csv
item_id,title
1,The Matrix
2,Inception
```

Add ratings using the same columns. Each user-item pair must be unique, and
ratings must be between 1 and 5.

## Notes and limitations

This is a small learning project with synthetic sample data. It demonstrates
the requested methods but is not a production recommendation service.
A new user must rate at least one item already present in the dataset.
