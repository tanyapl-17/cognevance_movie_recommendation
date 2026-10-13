
import pandas as pd

# Load the existing datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Create a table of users and their movie ratings
movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

# Calculate similarity between users
user_similarity = movie_matrix.T.corr()

def recommend_movies(user_id, number_of_recommendations=10):
    if user_id not in movie_matrix.index:
        print("User ID not found in the ratings dataset.")
        return

    # Find users with similar movie preferences
    similar_users = user_similarity[user_id].drop(
        index=user_id
    ).sort_values(ascending=False)

    # Select the 5 most similar users
    similar_users = similar_users.head(5)

    # Get movies already rated by the selected user
    already_rated = movie_matrix.loc[user_id].dropna().index

    # Calculate average ratings from similar users
    recommendations = movie_matrix.loc[
        similar_users.index
    ].mean().drop(index=already_rated, errors="ignore")

    # Select the highest-rated movies
    recommendations = recommendations.sort_values(
        ascending=False
    ).head(number_of_recommendations)

    # Display movie titles
    result = movies[movies["movieId"].isin(
        recommendations.index
    )][["movieId", "title", "genres"]]

    print(f"\nMovie recommendations for User {user_id}:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    recommend_movies(1)
