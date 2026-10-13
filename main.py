
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load the datasets
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

print("===== DATASET INFORMATION =====")
print("Total movies:", len(movies))
print("Total ratings:", len(ratings))
print("Unique users:", ratings["userId"].nunique())

# Step 2: Check missing values
print("\n===== MISSING VALUES =====")
print("Movies:\n", movies.isnull().sum())
print("Ratings:\n", ratings.isnull().sum())

# Step 3: Check duplicate rows
print("\n===== DUPLICATE ROWS =====")
print("Duplicate movies:", movies.duplicated().sum())
print("Duplicate ratings:", ratings.duplicated().sum())

# Step 4: Clean the data
movies = movies.drop_duplicates().copy()
ratings = ratings.drop_duplicates().copy()

movies = movies.dropna(subset=["movieId", "title"])
ratings = ratings.dropna(
    subset=["userId", "movieId", "rating"]
)

# Step 5: Analyze rating distribution
print("\n===== RATING DISTRIBUTION =====")
print(ratings["rating"].value_counts().sort_index())

ratings["rating"].value_counts().sort_index().plot(
    kind="bar",
    title="Distribution of Movie Ratings",
    xlabel="Rating",
    ylabel="Number of Ratings"
)

plt.tight_layout()
plt.show()

# Step 6: Find the 10 most-rated movies
movie_stats = ratings.groupby("movieId").agg(
    number_of_ratings=("rating", "count"),
    average_rating=("rating", "mean")
).reset_index()

movie_stats = movie_stats.merge(
    movies[["movieId", "title"]],
    on="movieId",
    how="left"
)

top_movies = movie_stats.sort_values(
    "number_of_ratings",
    ascending=False
).head(10)

print("\n===== TOP 10 MOST-RATED MOVIES =====")
print(top_movies[["title", "number_of_ratings",
                  "average_rating"]].to_string(index=False))

top_movies.plot(
    x="title",
    y="number_of_ratings",
    kind="bar",
    legend=False,
    title="Top 10 Most-Rated Movies"
)

plt.xlabel("Movie")
plt.ylabel("Number of Ratings")
plt.xticks(rotation=75, ha="right")
plt.tight_layout()
plt.savefig("outputs/top_10_movies.png", bbox_inches="tight")
plt.close()

# Save the rating distribution chart
plt.figure()
ratings["rating"].value_counts().sort_index().plot(
    kind="bar",
    title="Distribution of Movie Ratings",
    xlabel="Rating",
    ylabel="Number of Ratings"
)
plt.tight_layout()
plt.savefig("outputs/rating_distribution.png", bbox_inches="tight")
plt.close()


print("\nData analysis completed successfully!")


# Step 7: Create a movie recommendation function

def recommend_movies(movie_title, top_n=10):
    # Find the movie ID
    movie_match = movies[
        movies["title"].str.contains(
            movie_title, case=False, na=False
        )
    ]

    if movie_match.empty:
        print("Movie not found!")
        return

    movie_id = movie_match.iloc[0]["movieId"]

    # Get users who rated this movie
    users = ratings[
        ratings["movieId"] == movie_id
    ]["userId"]

    # Find other movies rated by the same users
    similar_ratings = ratings[
        ratings["userId"].isin(users)
    ]

    # Calculate average ratings and rating counts
    recommendations = (
        similar_ratings.groupby("movieId")
        .agg(
            average_rating=("rating", "mean"),
            number_of_ratings=("rating", "count")
        )
        .reset_index()
    )

    # Exclude the selected movie
    recommendations = recommendations[
        recommendations["movieId"] != movie_id
    ]

    # Keep movies with at least 20 ratings
    recommendations = recommendations[
        recommendations["number_of_ratings"] >= 20
    ]

    # Sort by average rating
    recommendations = recommendations.sort_values(
        by=["average_rating", "number_of_ratings"],
        ascending=False
    ).head(top_n)

    # Add movie titles
    recommendations = recommendations.merge(
        movies[["movieId", "title"]],
        on="movieId",
        how="left"
    )

    print(f"\nMovies recommended for: {movie_match.iloc[0]['title']}")
    print(
        recommendations[
            ["title", "average_rating", "number_of_ratings"]
        ].to_string(index=False)
    )


# Test the recommendation function
recommend_movies("Toy Story")
