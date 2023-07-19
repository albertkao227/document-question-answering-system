import os
import requests
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
api_key = os.getenv("OPENAI_API_KEY")




@app.route("/")
def index():
    return render_template("index.html")



@app.route("/results", methods=["POST"])
def answer():
    topic = request.form["topic"]
    prompt = request.form["prompt"]
    model = 'gpt-3.5-turbo'
    completions = requests.post(
    'https://api.openai.com/v1/completions',
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    },
    json = {
        'model': model,
        'prompt': 'what is your name',
        'temperature': 0.4,
        'max_tokens': 300
    })
    print(completions)
    return render_template("results.html", response=completions.json()) 



 

