from flask import Flask, request, render_template
import json
import requests
import socket
import time
import src.random_model as rm
from pickle import dump
import pickle
from datetime import datetime
from pymongo import MongoClient
import pandas as pd
import yaml # to load the string as a dictionary

app = Flask(__name__)
PORT = 5353
REGISTER_URL = "http://10.5.83.19:5000/register"
DATA = []
TIMESTAMP = []
client = MongoClient()
db = client.fraudstudy
collection = db.event

with open('static/model.pkl') as f:
    model = pickle.load(f)

with open('static/model_tfid.pkl') as f:
    model_tfidf = pickle.load(f)

@app.route('/', methods=['GET'])
def index():
    return render_template('front_end.html')

@app.route('/score', methods=['POST'])
def score():
    collection.insert_one(yaml.load(json.dumps(request.json, sort_keys=True, indent=None,
                       separators=(',', ': '))+'\n'))
    return ""


@app.route('/check')
def check():
    with open('data/predict_data.json', 'r+') as f:
        DATA = [line for line in f]
    with open('data/timestamp_data.json', 'r+') as f:
        TIMESTAMP = [float(line) for line in f]
    line1 = "Number of data points: {0}".format(len(DATA))
    if DATA and TIMESTAMP:
        dt = datetime.fromtimestamp(TIMESTAMP[-1])
        data_time = dt.strftime('%Y-%m-%d %H:%M:%S')
        line2 = "Latest datapoint received at: {0}".format(data_time)
        line3 = DATA[-1]
        output = "{0}\n\n{1}\n\n{2}".format(line1, line2, line3)
    else:
        output = line1
    return output, 200, {'Content-Type': 'text/css; charset=utf-8'}


@app.route('/main')
def main():
    word = 'Hello, there'
    return word, 200, {'Content-Type': 'text/css; charset=utf-8'}

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    client = MongoClient('mongodb://localhost:27017/')
    db = client.fraudstudy
    collection = db.event
    data = []
    allevents = collection.find().sort('_id',-1).limit(10)
    DATA = list(allevents)
    _ = map(lambda x : x.pop(u'_id', None), DATA)
    for datum in DATA:
        data.append(datum)
    ids = []
    for datum in data:
        ids.append(datum['object_id'])
    X = model.get_new_data(data)
    prediction1 = model.predict_proba(X)[:, 1]
    X_tfidf_df = pd.DataFrame(data)
    X_tfidf = X_tfidf_df['description']
    prediction2 = model_tfidf.predict_proba(X_tfidf)
    preds = zip(prediction1, prediction2)
    prediction = [max(x1, x2) for x1, x2 in preds]
    color = []
    for i, pred in enumerate(prediction):
        if pred > 0.8:
            prediction[i] = 3
            color.append('bgcolor="red">')
        elif pred > 0.5:
            prediction[i] = 2
            color.append('bgcolor="orange">')
        elif pred > 0.2:
            prediction[i] = 1
            color.append('bgcolor="yellow">')
        else:
            prediction[i] = 0
            color.append('bgcolor="lightgreen">')
    ids = map(str, ids)
    prediction = map(str, prediction)
    output = zip(ids, prediction)
    html_string = '<table><tr><th>ID</th><th>Fraud Level</th></tr>'
    for i, pair in enumerate(output):
        html_string += '<tr><td ' + color[i] + pair[0] + '</td><td ' + color[i] + pair[1] + '</td></tr>'
    html_string += '</table>'
    #pred = '\n'.join(output)
    return html_string


def register_for_ping(ip, port):
    registration_data = {'ip': ip, 'port': port}
    requests.post(REGISTER_URL, data=registration_data)


if __name__ == '__main__':
    # Register for pinging service
    ip_address = '' #INSERT YOUR IP ADDRESS HERE
    print "attempting to register %s:%d" % (ip_address, PORT)
    register_for_ping(ip_address, str(PORT))

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
