from flask import Flask, render_template, request
import requests
import pickle


app = Flask(__name__)

lr = pickle.load(open('lr.pkl', 'rb'))
cv = pickle.load(open('cv.pkl', 'rb'))

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        text=request.form['text']
        normalized=cv.transform([text])
        result=lr.predict(normalized)
        s=""
        if(result==1):
            s="Positive Review"
        else:
            s="Negative Review"
    return render_template("index.html", s=s)

if __name__=="__main__":
    app.run(debug=True)
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    