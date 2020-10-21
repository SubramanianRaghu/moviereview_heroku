from flask import Flask,render_template,request
import pickle

filename = 'model_moviewreview.pkl'
classifier = pickle.load(open(filename, 'rb'))
with open('cv-transform.pkl','rb') as fin:
    cv = pickle.load(fin)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
        return render_template('result.html', prediction = my_prediction)
    
if __name__ == '__main__':
    app.run(debug=True)
