## Import libraries ##
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

#function for preprocessing data 
def preprocess_data(df, is_train=True, frequency_map=None):
    #drop columns that have been tested for correlation and test data performance (they tend to cause overfitting, or are irrelevant)
    df.drop(['movie_title', 'plot_keywords', 'title_embedding', 'id'], axis=1, inplace=True)
    
    #binary encode the film genres
    unique_genres = set()
    df['genres'].str.split('|').apply(unique_genres.update)
    unique_genres = list(unique_genres)
    for genre in unique_genres:
        df[genre] = df['genres'].apply(lambda x: 1 if genre in x.split('|') else 0)
    df.drop('genres', axis=1, inplace=True)
    
    #create a binary feature for director_facebook_likes
    df['director_has_facebook'] = df['director_facebook_likes'].apply(lambda x: 1 if x > 0 else 0)
    #replace 'assumed' missing values with median values for movie_facebook_likes and director_facebook_likes
    df['movie_facebook_likes'] = df['movie_facebook_likes'].replace(0, df['movie_facebook_likes'].median())
    df['director_facebook_likes'] = df['director_facebook_likes'].replace(0, df['director_facebook_likes'].median())
    
    #apply one-hot encoding for low-dimensional categorical features
    df = pd.get_dummies(df, columns=['language', 'country', 'content_rating'], drop_first=True)

    #apply frequency encoding for highly dimensional categorical features
    high_dimensional_features = ['director_name', 'actor_2_name', 'actor_1_name', 'actor_3_name']
    if is_train:
        frequency_map = {}
        for feature in high_dimensional_features:
            frequency_map[feature] = df[feature].value_counts().to_dict()
            df[feature] = df[feature].map(frequency_map[feature])
    else:
        for feature in high_dimensional_features:
            df[feature] = df[feature].map(frequency_map[feature]).fillna(0)
    
    return df, frequency_map


## Load data ##
#update path to relevant path for datasets
train_data_path = 'c:/Users/menez/OneDrive/Unimelb/2024/COMP30027/Assignment2/train_dataset.csv'
test_data_path = 'c:/Users/menez/OneDrive/Unimelb/2024/COMP30027/Assignment2/test_dataset.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

test_ids = test_df[['id']]

## Preprocess Data ##
#preprocess train and test data (use same frequency map for target encoding)
encoded_train_df, frequency_map = preprocess_data(train_df)
encoded_test_df, _ = preprocess_data(test_df, is_train=False, frequency_map=frequency_map)
encoded_test_df = encoded_test_df.reindex(columns=encoded_train_df.columns, fill_value=0)

#remove the target column from test data features from reindexing columns in pre-processing
if 'imdb_score_binned' in encoded_test_df.columns:
    encoded_test_df.drop(columns=['imdb_score_binned'], inplace=True)

#scale numerical features
numerical_features = ['num_critic_for_reviews', 'duration', 'director_facebook_likes', 'actor_3_facebook_likes',
                      'actor_1_facebook_likes', 'gross', 'num_voted_users', 'cast_total_facebook_likes',
                      'num_user_for_reviews', 'title_year', 'actor_2_facebook_likes', 'movie_facebook_likes',
                      'average_degree_centrality'] + list(frequency_map.keys())

scale_method = MinMaxScaler()
encoded_train_df[numerical_features] = scale_method.fit_transform(encoded_train_df[numerical_features])
encoded_test_df[numerical_features] = scale_method.transform(encoded_test_df[numerical_features])

## Train Models ##
#split data into train and test for model selection and evaluatoon (HHGTTG)
X = encoded_train_df.drop(columns=['imdb_score_binned'])
y = encoded_train_df['imdb_score_binned']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "ZeroR Baseline": DummyClassifier(strategy='most_frequent'),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, C=0.1),  
    "XGBoost": XGBClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    print(f"{name}")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print(classification_report(y_val, y_pred))

## Hyperparameter Tuning ##
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
dt_params = {
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
xgb_params = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

#gridsearch for best hyperparameters for each model
rf_grid_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, scoring='accuracy')
rf_grid_search.fit(X_train, y_train)
print("Best parameters for Random Forest:", rf_grid_search.best_params_)
print("Best cross-validation accuracy for Random Forest:", rf_grid_search.best_score_)

dt_grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='accuracy')
dt_grid_search.fit(X_train, y_train)
print("Best parameters for Decision Tree:", dt_grid_search.best_params_)
print("Best cross-validation accuracy for Decision Tree:", dt_grid_search.best_score_)

xgb_grid_search = GridSearchCV(XGBClassifier(random_state=42), xgb_params, cv=5, scoring='accuracy')
xgb_grid_search.fit(X_train, y_train)
print("Best parameters for XGBoost:", xgb_grid_search.best_params_)
print("Best cross-validation accuracy for XGBoost:", xgb_grid_search.best_score_)

#re-train final models with optimal hyperparameters
final_decision_tree = DecisionTreeClassifier(**dt_grid_search.best_params_, random_state=42)
final_decision_tree.fit(X_train, y_train)
y_pred_final_dt = final_decision_tree.predict(X_val)
print("Final Decision Tree Classifier")
print("Accuracy:", accuracy_score(y_val, y_pred_final_dt))
print(classification_report(y_val, y_pred_final_dt))

final_random_forest = RandomForestClassifier(**rf_grid_search.best_params_, random_state=42)
final_random_forest.fit(X_train, y_train)
y_pred_final_rf = final_random_forest.predict(X_val)
print("Final Random Forest Classifier")
print("Accuracy:", accuracy_score(y_val, y_pred_final_rf))
print(classification_report(y_val, y_pred_final_rf))

final_xgboost = XGBClassifier(**xgb_grid_search.best_params_, random_state=42)
final_xgboost.fit(X_train, y_train)
y_pred_final_xgb = final_xgboost.predict(X_val)
print("Final XGBoost Classifier")
print("Accuracy:", accuracy_score(y_val, y_pred_final_xgb))
print(classification_report(y_val, y_pred_final_xgb))

#train LR with adjustments
adjusted_logistic_regression = LogisticRegression(max_iter=10000, solver='saga', random_state=42, C=0.1)
adjusted_logistic_regression.fit(X_train, y_train)
y_pred_final_lr = adjusted_logistic_regression.predict(X_val)
print("Adjusted Logistic Regression Model")
print("Accuracy:", accuracy_score(y_val, y_pred_final_lr))
print(classification_report(y_val, y_pred_final_lr))

#select the best model based on validation accuracy
best_model, best_accuracy = max([(final_random_forest, accuracy_score(y_val, y_pred_final_rf)), 
                                 (final_xgboost, accuracy_score(y_val, y_pred_final_xgb)), 
                                 (adjusted_logistic_regression, accuracy_score(y_val, y_pred_final_lr))], 
                                key=lambda x: x[1])

print(f"Selected Best Model: {best_model.__class__.__name__} with accuracy: {best_accuracy}")

## Final Prediction ##
#train final model on full training data and classify
##override best_model to xgboost based on kaggle test performance
best_model = final_xgboost
best_model.fit(X, y)
test_predictions = best_model.predict(encoded_test_df)

#output submission csv
output = pd.DataFrame({
    'id': test_ids['id'],
    'imdb_score_binned': test_predictions
})
output.to_csv('A_Result_final_xg.csv', index=False)
print("!successful classification!")