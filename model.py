import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV
from vectorization import vectorize_data, tfidf_vectorize

# Load the data
train = ...
test = ...

# Vectorize the data
train, test = vectorize_data(train, test)
train, test = tfidf_vectorize(train, test)

# Initialize and fit the model
features_to_drop = ['id', 'keyword','text','clean_text', 'hashtags', 'mentions','links']
scaler = MinMaxScaler()
X_train = train.drop(columns=features_to_drop + ['target'])
y_train = train['target']
X_test = test.drop(columns=features_to_drop)

# Create and fit the pipeline
lr = LogisticRegression(solver='liblinear', random_state=777)
pipeline = Pipeline([('scale',scaler), ('lr', lr)])
pipeline.fit(X_train, y_train)

# Evaluate the model
print('Training accuracy: %.4f and Training f-1 score: %.4f' %
(pipeline.score(X_train, y_train), f1_score(y_train, pipeline.predict(X_train))))

# Perform cross-validation
cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=123)
cv_score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
print('Cross validation F-1 score: %.3f' %np.mean(cv_score))

# Grid search for optimal parameters
grid = {"C":np.logspace(-2,2,5), "penalty":["l1","l2"]}
lr_cv = GridSearchCV(LogisticRegression(solver='liblinear', random_state=20), grid, cv=cv, scoring = 'f1')

pipeline_grid = Pipeline([('scale',scaler), ('gridsearch', lr_cv),])
pipeline_grid.fit(X_train, y_train)

print("Best parameter: ", lr_cv.best_params_)
print("F-1 score: %.3f" %lr_cv.best_score_)
