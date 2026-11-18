"""Collaborative-filtering recommender built with the Surprise library."""
from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Iterable

import pandas as pd
from surprise import Dataset, KNNBasic, Reader


class CollaborativeFilteringRecommender:
    """
    User-based and item-based collaborative filtering.

    user_knn recommends for existing users. item_knn supplies item-item
    similarities so a new user can be represented by a short rating profile.
    """

    def __init__(self, ratings: pd.DataFrame, movies: pd.DataFrame | None = None):
        required = {"user_id", "item_id", "rating"}
        missing = required.difference(ratings.columns)
        if missing:
            raise ValueError(f"ratings is missing columns: {sorted(missing)}")
        if ratings.empty:
            raise ValueError("ratings cannot be empty")

        self.ratings = ratings.loc[:, ["user_id", "item_id", "rating"]].copy()
        self.ratings["user_id"] = self.ratings["user_id"].astype(str)
        self.ratings["item_id"] = self.ratings["item_id"].astype(str)
        self.ratings["rating"] = pd.to_numeric(self.ratings["rating"], errors="raise")
        if not self.ratings["rating"].between(1, 5).all():
            raise ValueError("Every rating must be between 1 and 5.")
        if self.ratings.duplicated(["user_id", "item_id"]).any():
            raise ValueError("Each user-item pair must have only one rating.")

        if movies is None:
            self.movies = pd.DataFrame(columns=["item_id", "title"])
        else:
            if not {"item_id", "title"}.issubset(movies.columns):
                raise ValueError("movies must contain item_id and title columns.")
            self.movies = movies.loc[:, ["item_id", "title"]].copy()
            self.movies["item_id"] = self.movies["item_id"].astype(str)
            self.movies["title"] = self.movies["title"].astype(str)

        self.user_knn = None
        self.item_knn = None
        self.trainset = None
        self.item_trainset = None

    def fit(self) -> "CollaborativeFilteringRecommender":
        """Train separate user-user and item-item cosine KNN models."""
        reader = Reader(rating_scale=(1, 5))
        dataset = Dataset.load_from_df(
            self.ratings[["user_id", "item_id", "rating"]], reader
        )
        self.trainset = dataset.build_full_trainset()

        # User-based model: compare users and predict ratings for known users.
        self.user_knn = KNNBasic(
            k=20,
            min_k=1,
            sim_options={"name": "cosine", "user_based": True, "min_support": 1},
            verbose=False,
        )
        self.user_knn.fit(self.trainset)

        # Item-based model: compare items; used to score a new user's rated items.
        self.item_knn = KNNBasic(
            k=20,
            min_k=1,
            sim_options={"name": "cosine", "user_based": False, "min_support": 1},
            verbose=False,
        )
        self.item_knn.fit(self.trainset)
        self.item_trainset = self.item_knn.trainset
        return self

    def _ensure_fitted(self) -> None:
        if self.user_knn is None or self.item_knn is None:
            raise RuntimeError("The recommender is not trained. Call fit() first.")

    def _get_title(self, item_id: str) -> str:
        """Look up a readable title; warn and return the ID if no title exists."""
        match = self.movies.loc[self.movies["item_id"] == str(item_id), "title"]
        if match.empty:
            warnings.warn(f"No movie title found for item_id={item_id!r}.", stacklevel=2)
            return str(item_id)
        return str(match.iloc[0])

    def recommend(self, user_id: str, top_n: int = 5) -> list[dict]:
        """Return top-N unseen items for a user already in the training data."""
        self._ensure_fitted()
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")

        uid = str(user_id)
        if uid not in set(self.ratings["user_id"]):
            raise ValueError(
                f"Unknown user {uid!r}; use recommend_for_new_user() instead."
            )

        already_rated = set(
            self.ratings.loc[self.ratings["user_id"] == uid, "item_id"]
        )
        all_items = sorted(set(self.ratings["item_id"]))
        unseen_items = [iid for iid in all_items if iid not in already_rated]

        predictions = [
            (iid, float(self.user_knn.predict(uid, iid).est))
            for iid in unseen_items
        ]
        predictions.sort(key=lambda row: row[1], reverse=True)

        return [
            {
                "item_id": iid,
                "title": self._get_title(iid),
                "estimated_rating": round(score, 3),
            }
            for iid, score in predictions[:top_n]
        ]

    def user_similarity(self, user_id: str, top_n: int = 5) -> list[dict]:
        """Return the most similar training users and cosine similarity scores."""
        self._ensure_fitted()
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        try:
            inner_uid = self.trainset.to_inner_uid(str(user_id))
        except ValueError:
            return []

        similarities = self.user_knn.sim[inner_uid]
        neighbors = [
            (self.trainset.to_raw_uid(index), float(score))
            for index, score in enumerate(similarities)
            if index != inner_uid
        ]
        neighbors.sort(key=lambda row: row[1], reverse=True)
        return [
            {"user_id": uid, "similarity": round(score, 4)}
            for uid, score in neighbors[:top_n]
        ]

    def recommend_for_new_user(
        self, user_ratings: dict[str, float], top_n: int = 5
    ) -> list[dict]:
        """
        Recommend for a user not present in training data.

        user_ratings is {known_item_id: rating}. Each unseen item's score is the
        similarity-weighted average of the new user's ratings for known items.
        """
        self._ensure_fitted()
        if top_n < 1:
            raise ValueError("top_n must be at least 1.")
        if not user_ratings:
            raise ValueError("Provide at least one rating for a known item.")

        profile = {str(item): float(rating) for item, rating in user_ratings.items()}
        if any(not 1 <= rating <= 5 for rating in profile.values()):
            raise ValueError("Ratings must be between 1 and 5.")

        known_items = set(self.ratings["item_id"])
        unknown_items = set(profile).difference(known_items)
        if unknown_items:
            raise ValueError(f"Unknown item IDs: {sorted(unknown_items)}")

        candidates = sorted(known_items.difference(profile))
        global_mean = float(self.ratings["rating"].mean())
        scored_items = []

        for candidate in candidates:
            candidate_inner = self.item_trainset.to_inner_iid(candidate)
            weighted_total = 0.0
            similarity_total = 0.0

            for rated_item, rating in profile.items():
                rated_inner = self.item_trainset.to_inner_iid(rated_item)
                similarity = float(
                    self.item_knn.sim[candidate_inner][rated_inner]
                )
                if similarity > 0:
                    weighted_total += similarity * rating
                    similarity_total += similarity

            estimated = (
                weighted_total / similarity_total
                if similarity_total > 0
                else global_mean
            )
            scored_items.append((candidate, estimated))

        scored_items.sort(key=lambda row: row[1], reverse=True)
        return [
            {
                "item_id": iid,
                "title": self._get_title(iid),
                "estimated_rating": round(score, 3),
            }
            for iid, score in scored_items[:top_n]
        ]


def precision_recall_at_k(
    predictions: Iterable, k: int = 5, threshold: float = 4.0
) -> dict:
    """
    Compute mean Precision@K and Recall@K from Surprise Prediction objects.

    A true rating >= threshold is considered relevant. Precision is hits / K;
    recall is hits / all relevant held-out items for that user.
    """
    if k < 1:
        raise ValueError("k must be at least 1.")

    user_predictions = defaultdict(list)
    for prediction in predictions:
        uid, _, true_rating, estimated_rating, _ = prediction
        user_predictions[uid].append((float(estimated_rating), float(true_rating)))

    precision_values = []
    recall_values = []

    for user_rows in user_predictions.values():
        user_rows.sort(key=lambda row: row[0], reverse=True)
        top_k = user_rows[:k]
        relevant_count = sum(true_rating >= threshold for _, true_rating in user_rows)
        hit_count = sum(true_rating >= threshold for _, true_rating in top_k)

        precision_values.append(hit_count / k)
        if relevant_count:
            recall_values.append(hit_count / relevant_count)

    return {
        "precision@k": round(
            sum(precision_values) / len(precision_values), 4
        ) if precision_values else 0.0,
        "recall@k": round(
            sum(recall_values) / len(recall_values), 4
        ) if recall_values else 0.0,
        "k": k,
        "relevance_threshold": threshold,
    }
