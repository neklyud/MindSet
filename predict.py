import preprocess
from sklearn.model_selection import cross_validate

def predict(X, model):
    return model.predict(X).ravel()

def get_score(X, y, model):
    scores = cross_validate(model, X, y, cv=5,
                            scoring=('accuracy'),
                            return_train_score=True)
    return scores['test_score'].mean()