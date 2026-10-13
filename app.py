
import streamlit as st
import pandas as pd

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

# -------------------------------
# CUSTOM DESIGN
# -------------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
h1 {
    color: #243b64;
}
.stButton > button {
    background-color: #315efb;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD DATA
# -------------------------------
@st.cache_data
def load_data():
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")

    movies = movies.drop_duplicates(subset="movieId")
    ratings = ratings.drop_duplicates()

    ratings = ratings.dropna(
        subset=["userId", "movieId", "rating"]
    )

    ratings["userId"] = ratings["userId"].astype(int)
    ratings["movieId"] = ratings["movieId"].astype(int)

    return movies, ratings


try:
    movies, ratings = load_data()
except FileNotFoundError:
    st.error(
        "Dataset files not found. Please keep movies.csv and "
        "ratings.csv in the same folder as app.py."
    )
    st.stop()

# -------------------------------
# HEADER
# -------------------------------
st.title("🎬 Movie Recommendation System")
st.write(
    "Discover movies you may enjoy using collaborative filtering "
    "and ratings from similar users."
)

st.divider()

# -------------------------------
# DASHBOARD STATISTICS
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric(
    "🎞️ Total Movies",
    f"{movies['movieId'].nunique():,}"
)

col2.metric(
    "👥 Total Users",
    f"{ratings['userId'].nunique():,}"
)

col3.metric(
    "⭐ Total Ratings",
    f"{len(ratings):,}"
)

st.divider()

# -------------------------------
# USER SELECTION
# -------------------------------
st.subheader("Find Your Recommendations")

user_ids = sorted(ratings["userId"].unique())

if len(user_ids) < 2:
    st.warning("At least two users are needed for recommendations.")
    st.stop()

col1, col2 = st.columns(2)

with col1:
    user_id = st.selectbox(
        "Select User ID",
        user_ids
    )

with col2:
    number_of_movies = st.slider(
        "Number of recommendations",
        min_value=5,
        max_value=15,
        value=10
    )

# -------------------------------
# RECOMMENDATION SYSTEM
# -------------------------------
if st.button("🎯 Recommend Movies"):

    with st.spinner("Finding movies you may like..."):

        movie_matrix = ratings.pivot_table(
            index="userId",
            columns="movieId",
            values="rating"
        )

        # Calculate similarity between users
        user_similarity = movie_matrix.T.corr()

        # Find similar users
        similar_users = (
            user_similarity[user_id]
            .drop(index=user_id, errors="ignore")
            .dropna()
            .sort_values(ascending=False)
            .head(5)
        )

        # Movies already watched by the selected user
        already_rated = (
            movie_matrix.loc[user_id]
            .dropna()
            .index
        )

        if similar_users.empty:

            st.warning(
                "No similar users found. Try another User ID."
            )

        else:

            # Calculate average ratings from similar users
            recommendations = (
                movie_matrix.loc[similar_users.index]
                .mean()
                .drop(index=already_rated, errors="ignore")
                .dropna()
                .sort_values(ascending=False)
                .head(number_of_movies)
            )

            if recommendations.empty:

                st.warning(
                    "No new recommendations found for this user."
                )

            else:

                # Create result table in recommendation order
                results = pd.DataFrame({
                    "movieId": recommendations.index
                })

                results["Predicted Rating"] = (
                    recommendations.values
                )

                results = results.merge(
                    movies[["movieId", "title", "genres"]],
                    on="movieId",
                    how="left"
                )

                results = results[
                    ["title", "genres", "Predicted Rating"]
                ]

                results["Predicted Rating"] = (
                    results["Predicted Rating"].round(2)
                )

                results = results.rename(columns={
                    "title": "Movie Title",
                    "genres": "Genres"
                })

                # Display results
                st.success("Recommendations generated successfully!")
                st.subheader("🍿 Recommended Movies")

                st.dataframe(
                    results.reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True
                )

                # Download recommendations
                csv_data = results.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Recommendations (CSV)",
                    data=csv_data,
                    file_name=f"movie_recommendations_user_{user_id}.csv",
                    mime="text/csv"
                )

# -------------------------------
# FOOTER
# -------------------------------
st.divider()

st.caption(
    "Built with Python, Pandas and Streamlit | "
    "Recommendation technique: Collaborative Filtering"
)
