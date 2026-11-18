
from models.logistic_regression_model import build_model as logistic_regression
from models.random_forest_model import build_model as random_forest
from models.extra_trees_model import build_model as extra_trees
from models.gradient_boosting_model import build_model as gradient_boosting
from models.sgd_model import build_model as sgd


def build_models():
    return {
        "Logistic Regression": logistic_regression(),
        "Random Forest": random_forest(),
        "Extra Trees": extra_trees(),
        "Hist Gradient Boosting": gradient_boosting(),
        "SGD Classifier": sgd()
    }
