# Movie Recommendation Engine

This is a simple Flask-based movie recommendation engine that allows users to input a movie name, provide a rating, and enter a natural language prompt to get hybrid machine learning-based movie recommendations boosted by the prompt.

## Features

- User enters a movie name (free text) and rates it (1–5 scale).
- User provides a natural language recommendation prompt.
- The system generates hybrid (Collaborative Filtering + Content-Based) recommendations.
- Recommendations are boosted based on the user's prompt keywords.

## Sample Dataset

The project includes a small sample dataset of movies with genres and tags, along with some community user ratings for collaborative filtering.

## Requirements

- Python 3.x
- Flask

## Installation and Running

1. Install Flask if you don't have it already:

pip install flask

text

2. Run the Flask application:

python app.py

text

3. Open your browser and navigate to:

[http://127.0.0.1:5000](http://127.0.0.1:5000)

4. Enter a movie title from the dataset (e.g. "Inception"), provide a rating from 1 to 5, and optionally enter a recommendation prompt. Submit to get recommendations.

## How It Works

- The app uses cosine similarity to measure similarity between user ratings and movie content.
- Collaborative Filtering (CF) predicts ratings based on similar users.
- Content-Based (CB) filtering builds user profiles from liked movie features.
- Hybrid scoring combines CF and CB recommendations.
- Prompt-based boosting increases scores of movies matching the prompt keywords.

## Project Structure

- `app.py` (or `movie.py`): Main Flask application and recommendation engine code.
- Movie dataset and sample community data are defined within the script.

## Notes

- This is a minimal prototype with a small hardcoded dataset for demonstration.
- You can extend the dataset or integrate with an external movie database API.
- The recommendation logic can be further improved or tuned based on real-world data.

## Contact

For questions or contributions, please email me at harshal27patel@gmail.com

---

Enjoy exploring movie recommendations with this simple yet flexible system!
