from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

app = Flask(__name__)
CORS(app)
# Load the trained model
with open('C:\\Users\\2003v\\OneDrive\\Desktop\\HTML\\webapp\\model\\model_mn.pkl', 'rb') as file:
    model = pickle.load(file)

with open('C:\\Users\\2003v\\OneDrive\\Desktop\\HTML\\webapp\\model\\count_vectorizer.pkl', 'rb') as file:
    count_vectorizer = pickle.load(file)

# NLTK setup
tokenizer = RegexpTokenizer(r"\w+")
en_stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

# CountVectorizer setup
cv = CountVectorizer(ngram_range=(1, 2))


def get_cleaned_text(text):
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    new_tokens = [token for token in tokens if token not in en_stopwords]
    stemmed_tokens = [ps.stem(token) for token in new_tokens]
    clean_text = " ".join(stemmed_tokens)
    return clean_text


@app.route('/')
def home():
    return render_template('main.html')


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.json['text']

        # Clean the text
        cleaned_text = get_cleaned_text(text)

        # Transform the cleaned text using the loaded CountVectorizer
        vectorized_text = count_vectorizer.transform([cleaned_text]).toarray()

        # Make predictions using the loaded model
        prediction = model.predict(vectorized_text)[0]

        return jsonify({'sentiment': prediction})



if __name__ == '__main__':
    app.run(debug=True)
