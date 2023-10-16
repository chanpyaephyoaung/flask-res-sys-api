from flask import Flask, request, jsonify
import json
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load Movies Metadata
metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

# Print the first three rows
# print(metadata['overview'].head())

# #Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN with an empty string
metadata['overview'] = metadata['overview'].fillna('')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(metadata['overview'])

# Output the shape of tfidf_matrix
# print(tfidf_matrix.shape)

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# print(cosine_sim.shape)

# Construct a reverse map of indices and movie titles
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

# print(indices[:10])

# Recommender system function

def get_recommendations(title, cosine_sim=cosine_sim):
   # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
   #  return metadata['title'].iloc[movie_indices]

   # Create a list of dictionaries with movie titles and similarity scores
    recommendations = [{'title': metadata['title'].iloc[i], 'overview': metadata['overview'].iloc[i], 'score': sim_scores[j][1]} for j, i in enumerate(movie_indices)]
    
    return json.dumps(recommendations)


print(get_recommendations('Leaving Las Vegas'))

app = Flask(__name__)


@app.route("/get-rec/<movie>")
def get_rec(movie):
   recommendations = get_recommendations(movie)
   
   print(recommendations)
   print("THE TYPE IS ", type(recommendations))
   return recommendations

if __name__ == '__main__':
    app.run(debug=True)