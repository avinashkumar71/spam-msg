from flask import Flask, render_template, request
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

app = Flask(__name__)


# transform function
def transform(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return ' '.join(y)


tfid = pickle.load(open('vectorizer.pkl', 'rb'))

model = pickle.load(open('model.pkl', 'rb'))


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text-area')
    transform_text = transform(text)
    vector = tfid.transform([transform_text]).toarray()
    result = model.predict(vector)[0]
    if result==0:
        result = '0';
    else:
        result = '1';

    return render_template('index.html',result=result)


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
