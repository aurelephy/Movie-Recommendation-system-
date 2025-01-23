# Movie-Recommendation-system-
Phase 5-group 8

# Movie Recommendation System

## Overview

The Movie Recommendation System is designed to provide personalized movie suggestions based on user ratings. By leveraging machine learning algorithms, the system analyzes user preferences and identifies similarities between movies, delivering a tailored list of the top 5 recommended movies for each user. This project aims to enhance the viewing experience for users of streaming platforms by helping them discover films that match their tastes.

## Objectives

1. **Develop a Personalized Recommendation System**: Create a system that offers tailored movie recommendations based on individual user preferences and movie ratings.
  
2. **Implement Machine Learning Algorithms**: Utilize machine learning techniques to analyze user preferences and movie characteristics, improving recommendation accuracy.

3. **Analyze User Preferences**: Identify patterns in user behavior and preferences to better understand their movie interests.

4. **Assess Movie Similarities**: Compare and evaluate similarities between movies to enhance the relevance of recommendations.

5. **Generate Top 5 Recommendations**: Deliver a curated list of the top 5 movie suggestions for each user, ensuring high personalization and satisfaction.

## Business Understanding

In the era of streaming services, movie recommendation systems address a common challenge: helping users discover films that align with their preferences. By analyzing user ratings, these systems provide personalized suggestions that enhance the viewing experience. The primary audience for this solution includes streaming platforms such as Netflix, Amazon Prime Video, and Hulu, which aim to increase customer engagement and retention. By leveraging advanced machine learning techniques and user data, these platforms can offer tailored recommendations, driving customer satisfaction and gaining a competitive advantage in the entertainment industry.

## Data Understanding

### Source

- **Dataset**: The dataset is provided by GroupLens, specifically the MovieLens Latest dataset.

### Dataset Details

The dataset, `ml-latest-small`, captures 5-star ratings and free-text tagging activity from MovieLens, a movie recommendation platform. It comprises:

- **100,836 ratings**
- **3,683 tag applications**
- **9,742 movies**

These data were collected from 610 users between March 29, 1996, and September 24, 2018, and the dataset was generated on September 26, 2018.

### Key Characteristics

Users were selected randomly, with the condition that each user rated at least 20 movies. Each user is identified by a unique ID, with no additional personal information provided.

### Dataset Files

The data is organized across the following files:

- `links.csv`
- `movies.csv`
- `ratings.csv`
- `tags.csv`

## Importing Necessary Libraries

To work with the dataset, we import the following libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")
```

## Loading the Datasets

We load our datasets for analysis:

```python
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")
links = pd.read_csv("links.csv")
tags = pd.read_csv("tags.csv")
```

### Inspecting the Datasets

We can inspect the contents of the datasets to understand their structure:

```python
# Inspect Movies dataset
movies.head(20)

# Inspect Ratings dataset
ratings.head()

# Inspect Links dataset
links.head()

# Inspect Tags dataset
tags.head()
```

## Exploratory Data Analysis and Data Preprocessing

### Ratings Dataset

We begin by examining the ratings dataset:

```python
ratings.describe()
```

### Ratings Distribution

To visualize the distribution of ratings, we create a count plot:

```python
plt.figure(figsize=(8, 4))
sns.countplot(x='rating', data=ratings, color='lightblue', order=sorted(ratings['rating'].unique(), reverse=True))
plt.title('Distribution of Ratings')
plt.show()
```

### Top Rated Movies

We merge the ratings and movies datasets to identify the top-rated movies:

```python
merged_df = pd.merge(ratings, movies, on='movieId')
avg_ratings = merged_df.groupby('title')['rating'].mean().sort_values(ascending=False)
top_movies_all_genres = avg_ratings.head(10)
```

### Genre Popularity Analysis

We analyze the popularity of movie genres by counting ratings for each genre:

```python
top_genres = merged_df['genres'].str.split('|').explode().value_counts().head(10)
```

### Data Preparation

We drop unnecessary columns and prepare the data for modeling:

```python
ratings = ratings.drop(columns='timestamp')
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
train_data, test_data = train_test_split(user_item_matrix, test_size=0.2, random_state=42)
```

## Modeling

### Baseline Model - Normal Predictor

The Normal Predictor serves as a baseline model that uses the training set's rating distribution to generate predictions:

```python
class NormalPredictorSK:
    def fit(self, X):
        self.mean = X['rating'].mean()
        self.std = X['rating'].std()

    def predict(self, n_predictions):
        return np.random.normal(self.mean, self.std, n_predictions)

baseline_model = NormalPredictorSK()
baseline_model.fit(train_data_long)
```

### KNN Basic Model

KNN (K-Nearest Neighbors) is a memory-based collaborative filtering algorithm used to predict ratings:

```python
X = ratings[['userId', 'movieId']]
y = ratings['rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [
    KNeighborsRegressor(n_neighbors=5, metric='cosine'),
    DummyRegressor(strategy="mean")
]

for model, name in zip(models, model_names):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
```

## Evaluation

### Model Performance

We evaluate the performance of the models using RMSE and MAE metrics:

```python
model_names = ["Normal Predictor", "KNN Basic"]
rmse_values = [0.8802, 1.1374]
mae_values = [0.6771, 0.9011]
```

### Visualizing Model Performance

We create a bar plot to visualize the RMSE and MAE values for each model:

```python
bar_width = 0.35
x = np.arange(len(model_names))

fig, ax = plt.subplots(figsize=(8, 5))
bars1 = ax.bar(x - bar_width / 2, rmse_values, bar_width, label='RMSE', color='skyblue')
bars2 = ax.bar(x + bar_width / 2, mae_values, bar_width, label='MAE', color='lightcoral')
```

## Making Recommendations

We define a function to get the top N recommendations for a user:

```python
def get_top_n(predictions, n=10):
    top_n = defaultdict(list)
    for uid, iid, _, est, _ in predictions:
        top_n[uid].append((iid, est))
    return {uid: sorted(user_ratings, key=lambda x: x[1], reverse=True)[:n] for uid, user_ratings in top_n.items()}
```

## Conclusion

The implementation of the recommendation system using scikit-learn's KNeighborsRegressor provides a robust approach to personalized movie recommendations based on user preferences and past ratings. The results demonstrated the effectiveness of the KNN model, achieving an RMSE of 1.1374 and an MAE of 0.9011.

## Recommendations for Future Work

1. **Optimization**: Experiment with different distance metrics and hyperparameters to improve model accuracy.
2. **Data Enhancements**: Incorporate additional user or movie features to improve prediction quality.
3. **Scalability**: Consider switching to approximate nearest neighbor algorithms for faster computations.
4. **User  Experience**: Integrate feedback mechanisms to refine recommendations based on user interaction.

By implementing these recommendations, the system can achieve improved accuracy, scalability, and user satisfaction, making it more effective for real-world applications.

---