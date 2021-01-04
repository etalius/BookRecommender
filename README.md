# BookRecommender
A book recommendation system using the goodreads Kaggle dataset of 10,000 books. The final recommendation app can be found here: https://app-book-recs.herokuapp.com

## Introduction
This project includes a Jupyter Notebook containing data cleaning, exploratory analysis and content-based and collaborative filtering recommendation systems on the goodreads-10k data set. The Flask app contains implementations of collaborative filtering, content-based filtering and a hybrid approach for users to recieve recommendations.

## Files
**Book Recommendations.ipynb** Jupyter Notebook outlining the process of data cleaning, initial anaylysis and development of the recommendation systems.<br />
**model.py** File containing the implementation of the K-Nearest Neighbors model for the app, and saving the model and other data needed for app.py <br />
**app.py** Implementation of the Flask app <br />
**templates/** HTML templates <br />
**data/** Saved KNN model, original data and indices used in app.py <br />


