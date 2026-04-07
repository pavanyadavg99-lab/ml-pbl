import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("1. Downloading dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=columns)

# Map target classes to Star Ratings manually (since it's the target variable)
target_mapping = {'unacc': 1, 'acc': 3, 'good': 4, 'vgood': 5}
df['class'] = df['class'].map(target_mapping)

X = df.drop('class', axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Building ML Pipeline...")
# Define the exact order of categories for Ordinal Encoding
# This teaches the AI that "low" < "med" < "high"
categories = [
    ['low', 'med', 'high', 'vhigh'], # buying
    ['low', 'med', 'high', 'vhigh'], # maint
    ['2', '3', '4', '5more'],        # doors
    ['2', '4', 'more'],              # persons
    ['small', 'med', 'big'],         # lug_boot
    ['low', 'med', 'high']           # safety
]

features = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']

# Preprocessor transforms text into ordered numbers automatically
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(categories=categories), features)
    ])

# Pipeline bundles the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

print("3. Tuning Hyperparameters with GridSearchCV...")
# Test different combinations to find the absolute best model
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 15],
    'classifier__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters found: {grid_search.best_params_}")

print("4. Evaluating Best Model...")
y_pred = best_model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Save the full pipeline (preprocessor + model)
joblib.dump(best_model, 'model.pkl')
print("Advanced Pipeline saved successfully as 'model.pkl'")