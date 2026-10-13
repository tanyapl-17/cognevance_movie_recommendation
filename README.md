# 🎬 Movie Recommendation System

## Project Overview
The Movie Recommendation System recommends movies to users based on ratings given by other users with similar preferences. It uses collaborative filtering to generate personalized movie suggestions.

## Objectives
- Analyze movie ratings and user preferences.
- Clean and preprocess the dataset.
- Identify similar users using rating patterns.
- Recommend movies that the selected user has not rated.
- Display results through an interactive web application.

## Technologies Used
- Python
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Collaborative Filtering

## Features
- Interactive dashboard with movie, user, and rating statistics.
- User ID selection.
- Personalized movie recommendations.
- Predicted rating scores based on similar users' ratings.
- Download recommendations as a CSV file.
- Data visualization of movie ratings.

## Dataset
The project uses `movies.csv` and `ratings.csv`.

Keep the dataset files in the project folder alongside `app.py`.

## How to Run the Project

1. Install Python.
2. Open the project folder in VS Code.
3. Install the required libraries:

   ```bash
   python -m pip install pandas matplotlib scikit-learn streamlit
   ```

4. Start the application:

   ```bash
   python -m streamlit run app.py
   ```

5. Open the local URL displayed in the terminal.

## Project Structure

```text
cognevance_movie_recommendation/
├── app.py
├── main.py
├── recommender.py
├── movies.csv
├── ratings.csv
├── outputs/
│   ├── rating_distribution.png
│   └── top_10_movies.png
└── README.md
```

## Author
AI/ML Internship Project — Cognevance Technologies

## Conclusion
This project demonstrates data preprocessing, exploratory data analysis, user similarity analysis, collaborative filtering, and deployment of a simple interactive recommendation application.
