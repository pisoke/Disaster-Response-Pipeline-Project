from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from sqlalchemy import create_engine
import plotly
import plotly.express as px
import json

app = Flask(__name__)

# Load data and model
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)
model = joblib.load("../models/classifier.pkl")

# Define routes
@app.route('/')
@app.route('/index')
def index():
    # Render web page with data visuals
    return render_template('master.html')

@app.route('/go')
def go():
    # Use model to predict classifications for query
    query = request.args.get('query', '')
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
    return render_template('go.html', query=query, classification_result=classification_results)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3001, debug=True)
