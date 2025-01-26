from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# Load the pre-trained model and vectorizer
model = joblib.load("D:\\portifilo\\end_to_end\\Hate_speech_dection\\navie_baises.pkl")
bow = joblib.load("D:\\portifilo\\end_to_end\\Hate_speech_dection\\bag_of_words.pkl")

@app.route("/")
def home():
    return render_template('index.html', result=None)

@app.route("/submit-review", methods=['POST'])
def submit_review():
    if request.method == 'POST':
        review = request.form['review']

        # Data cleaning
        review = review.lower()  # Convert to lowercase
        review = re.sub(r"[^a-zA-Z0-9\s]", "", review)  # Remove special characters
        review = re.sub(r"\s+", " ", review).strip()  # Remove extra whitespace

        # Transform review using Bag of Words
        transformed_review = bow.transform([review])

        # Make prediction
        prediction = model.predict(transformed_review)[0]

        # Interpret prediction
        result = "Positive" if prediction == 1 else "Negitive"

        return render_template('index.html', result=result, review=review)

if __name__ == "__main__":
    app.run(debug=True)
