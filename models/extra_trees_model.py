
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier


def build_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", ExtraTreesClassifier(
            n_estimators=100,
            max_depth=20,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ])
