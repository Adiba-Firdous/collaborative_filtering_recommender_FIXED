"""Train/test evaluation for the collaborative-filtering recommender."""
import pandas as pd
from surprise import Dataset, KNNBasic, Reader
from surprise.model_selection import train_test_split

from src.recommender import precision_recall_at_k


def evaluate(
    ratings: pd.DataFrame, test_size: float = 0.25, k: int = 5,
    threshold: float = 4.0
) -> dict:
    """Train on a random split and report Precision@K and Recall@K."""
    required = {"user_id", "item_id", "rating"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing columns: {sorted(missing)}")

    frame = ratings[["user_id", "item_id", "rating"]].copy()
    frame["user_id"] = frame["user_id"].astype(str)
    frame["item_id"] = frame["item_id"].astype(str)

    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(frame, reader)
    trainset, testset = train_test_split(
        data, test_size=test_size, random_state=42
    )

    model = KNNBasic(
        k=20,
        min_k=1,
        sim_options={"name": "cosine", "user_based": True, "min_support": 1},
        verbose=False,
    )
    model.fit(trainset)
    predictions = model.test(testset)
    return precision_recall_at_k(predictions, k=k, threshold=threshold)
