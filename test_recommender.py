from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import evaluate
from src.recommender import CollaborativeFilteringRecommender, precision_recall_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture
def recommender():
    ratings = pd.read_csv(DATA / "ratings.csv")
    movies = pd.read_csv(DATA / "movies.csv")
    return CollaborativeFilteringRecommender(ratings, movies).fit()


def test_existing_user_recommendations_exclude_rated_items(recommender):
    results = recommender.recommend("1", top_n=4)
    rated = set(
        recommender.ratings.loc[
            recommender.ratings["user_id"] == "1", "item_id"
        ]
    )
    assert len(results) <= 4
    assert all(row["item_id"] not in rated for row in results)
    assert all({"item_id", "title", "estimated_rating"} <= row.keys() for row in results)


def test_new_user_recommendations_exclude_profile_items(recommender):
    profile = {"1": 5, "3": 4, "5": 2}
    results = recommender.recommend_for_new_user(profile, top_n=4)
    assert len(results) <= 4
    assert all(row["item_id"] not in profile for row in results)


def test_unknown_user_raises(recommender):
    with pytest.raises(ValueError):
        recommender.recommend("unknown-user")


def test_new_user_requires_known_item(recommender):
    with pytest.raises(ValueError):
        recommender.recommend_for_new_user({"not-an-item": 5})


def test_evaluation_returns_metrics():
    ratings = pd.read_csv(DATA / "ratings.csv")
    metrics = evaluate(ratings, k=5)
    assert "precision@k" in metrics
    assert "recall@k" in metrics


def test_precision_recall_helper_empty_input():
    assert precision_recall_at_k([], k=5)["precision@k"] == 0.0
