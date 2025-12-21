from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def build_pipeline(preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("model", LogisticRegression(max_iter=1000))
        ]
    )
    return pipeline
