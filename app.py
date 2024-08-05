from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
with open('model_tfidf.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user = data.get('user')
    num_recommendations = data.get('num_recommendations')
    
    # Assume that you have a function in your model that takes a user and number of recommendations as input
    recommendations = model.get_recommendations(user, int(num_recommendations))
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)
