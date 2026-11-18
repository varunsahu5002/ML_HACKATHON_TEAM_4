
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", HistGradientBoostingClassifier(
            max_iter=100,
            max_leaf_nodes=20,
            learning_rate=0.1,
            random_state=42
        ))
    ])
