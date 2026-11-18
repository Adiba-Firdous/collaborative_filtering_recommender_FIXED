"""Command-line demo for the collaborative-filtering recommender."""
from pathlib import Path

import pandas as pd

from src.evaluate import evaluate
from src.recommender import CollaborativeFilteringRecommender

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


def main() -> None:
    ratings = pd.read_csv(DATA_DIR / "ratings.csv")
    movies = pd.read_csv(DATA_DIR / "movies.csv")

    recommender = CollaborativeFilteringRecommender(ratings, movies).fit()

    user_id = "1"
    print(f"\nTop recommendations for existing user {user_id}:")
    recommendations = recommender.recommend(user_id, top_n=5)
    for row in recommendations:
        print(
            f"{row['title']:<25} "
            f"predicted rating: {row['estimated_rating']:.3f}"
        )

    print(f"\nUsers most similar to user {user_id}:")
    for row in recommender.user_similarity(user_id, top_n=5):
        print(f"User {row['user_id']}: similarity {row['similarity']:.4f}")

    new_user_profile = {"1": 5, "3": 4, "5": 2}
    print("\nRecommendations for a new user:")
    for row in recommender.recommend_for_new_user(new_user_profile, top_n=5):
        print(
            f"{row['title']:<25} "
            f"predicted rating: {row['estimated_rating']:.3f}"
        )

    print("\nOffline evaluation:")
    print(evaluate(ratings, k=5))


if __name__ == "__main__":
    main()
