import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from scipy import sparse
import json
from utils.utils import get_movie_name_match
from flask_bootstrap import Bootstrap
# pip install Flask-Bootstrap

app = Flask(__name__)
Bootstrap(app)

# *** START Load assets ***
# Load model
model = pickle.load(open('knnpickle_file.pkl', 'rb'))

# Load data recommendation
movie_user_matrix_sparse = sparse.load_npz("movie_user_matrix_sparse.npz")

# load movie -> id mapper
with open('movie_name_to_id.json') as json_file:
    movie_name_to_id = json.load(json_file)

# load id -> movie mapper
with open('id_to_movie_name.json') as json_file:
    id_to_movie_name = json.load(json_file)

# *** END Load assets ***



# *** START Flask server code ***
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():

    input = list(request.form.values())

    fav_movie = str(input[0])
    n_recommendations = int(input[1])

    top_match_tuple = get_movie_name_match(movie_name_to_id, fav_movie)

    if top_match_tuple == None:
        # Error - No movie found in database
        return render_template('index.html', error ='No movie match found')

    distances, indices = model.kneighbors(movie_user_matrix_sparse[top_match_tuple], n_neighbors=n_recommendations+1)

    # remove first recommendation because is the same movie
    distances, indices = distances[0][1:], indices[0][1:]

    recommendations = list(zip(indices, distances))
    recommendations = sorted(recommendations, key = lambda x: x[1])

    recommendations_text = [id_to_movie_name[str(movie_index[0])] for movie_index in recommendations]
    distances_text = [round((1 - distance) * 100, 1) for distance in distances]
    res = [[x, y] for x, y in zip(recommendations_text, distances_text)]

    return render_template('index.html', res = res)


if __name__ == "__main__":
    app.run(debug=True)