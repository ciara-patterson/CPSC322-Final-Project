from flask import Flask, jsonify, request
import pickle
import os

# Make a Flask app
app = Flask(__name__)

#  We need to add two routes (functions that handle requests)
#  one for the homepage
@app.route("/", methods=["GET"])
    # return content and a status code
def index():
    return "<h1>Welcome to Ciara and Kat's App</h1>", 200

# one for the /predict
@app.route("/predict", methods=["GET"])
def predict():
    infile = open('movies_tree.p', 'rb')
    myb = pickle.load(infile)
    infile.close()

    budget = request.args.get('budget', '')
    votes = request.args.get('votes', '')
    genre = request.args.get('genre', '')
    rating = request.args.get('rating', '')
    score = request.args.get('score', '')
    star = request.args.get('star', '')
    director = request.args.get('director', '')
    writer = request.args.get('writer', '')
    # profitted = request.args.get('profitted','')
    prediction = myb.predict([[int(budget), int(votes), genre, rating, int(score), star, director, writer]])

    if prediction is not None:
        result = {'prediction': prediction}
        return jsonify(result), 200
    else:
        return 'Error making prediction', 400


if __name__ == "__main__":
    # app.run(debug=True)

    # uncomment when done
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port = port) # TODO: set debug to False for production
    # by default, Flask runs on port 5000
