import pandas as pd
import string
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline


def compute_features(df):
    """
    Compute features all together, in real world ML training, that would be a lot more features inside this function.
    """
    # simple feature for review text on word counts
    df['review_text_count'] = df['review_text'].apply(lambda x: len(str(x).split()))
    # simple feature for product_title text on special char counts
    df['product_title_special_char_count'] = df['product_title'].apply(lambda x: len([char for char in str(x) if char in string.punctuation]))
    return df.drop(['review_text', 'product_title'], axis=1) # drop the original columns


# Product data
# Define column names for product df
column_names_product = ['id', 'category', 'product_title']

# Read the TSV files into Pandas DataFrames
df_product_0 = pd.read_csv('dataset/products-data-0.tsv', sep='\t', names=column_names_product)
df_product_1 = pd.read_csv('dataset/products-data-1.tsv', sep='\t', names=column_names_product)
df_product_2 = pd.read_csv('dataset/products-data-2.tsv', sep='\t', names=column_names_product)
df_product_3 = pd.read_csv('dataset/products-data-3.tsv', sep='\t', names=column_names_product)

# Concatenate the DataFrames
concatenated_product_df = pd.concat([df_product_0, df_product_1, df_product_2, df_product_3], ignore_index=True)

# Fix: preprocess for the target column after EDA
concatenated_product_df['category'] = concatenated_product_df['category'].replace({'Kitchen': 0, 'Ktchen': 0, 'Jewelry': 1})

# Review data
# Define column names for review df
column_names_reviews = ['id', 'rating', 'review_text']

# Fix : there are one set of data have different order of column names
column_names_reviews_diff = ['rating', 'id', 'review_text']

# Read the TSV files into Pandas DataFrames
df_reviews_0 = pd.read_csv('dataset/reviews-0.tsv', sep='\t', names=column_names_reviews)
df_reviews_1 = pd.read_csv('dataset/reviews-1.tsv', sep='\t', names=column_names_reviews)
df_reviews_2 = pd.read_csv('dataset/reviews-2.tsv', sep='\t', names=column_names_reviews_diff)
df_reviews_3 = pd.read_csv('dataset/reviews-3.tsv', sep='\t', names=column_names_reviews)

# Concatenate the DataFrames
concatenated_reviews_df = pd.concat([df_reviews_0, df_reviews_1, df_reviews_2, df_reviews_3], ignore_index=True)

# Inner join using id so that we have full data to process when we are having data both from review and product df
full_df = pd.merge(concatenated_product_df, concatenated_reviews_df, on='id', how='inner')

# Define random state and shuffle the data
random.seed(333) 
full_df_shuffled = full_df.sample(frac=1, random_state=333) 

# Define the feature and target columns, drop unecessary ones
X = full_df_shuffled.drop(['category', 'id'], axis=1)
y = full_df_shuffled['category']

# Training and test set split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

# Feature engineering: in real world, we would want to process X_train and X_test seperated in case there are some features need to utilize the encoder/scaler etc.
X_train = compute_features(X_train)
X_test = compute_features(X_test)

# Create a pipeline with StandardScaler and Logistic Regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(solver='liblinear'))
])

# Define the hyperparameter tuning grid
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'classifier__penalty': ['l1', 'l2']
}

# Specify multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision',
    'recall': 'recall',
    'f1': 'f1'
}

# Create a GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring=scoring, refit='f1')

# Fit the GridSearchCV on the training data
grid_search.fit(X_train, y_train)

# Get and print the best parameters from the grid search
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Predict on the test set using the best model
y_pred = grid_search.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print the scores
print(f'Final F1 score on test set is {f1:.2f}')
print(f'Final Precision on test set is: {precision:.2f}')
print(f'Final Recall on test set is: {recall:.2f}')
print(f'Final Accuracy on test set is: {accuracy:.2f}')

# Display the confusion matrix
print("Confusion Matrix:")
print(conf_matrix)