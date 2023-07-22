#!/usr/bin/python3

import os
import openai
import requests
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
#api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")



@app.route("/")
def index():
    return render_template("index.html")



@app.route("/answer", methods=["POST"])
def answer():
    question = request.form["question"]
    category = request.form["category"]
    
    prompt = f'''
    Please answer following question in details:
    {question}

    Answer: 
    '''

    model = 'gpt-3.5-turbo'

    response = openai.Completion.create(
        engine="text-davinci-002", prompt=f"Q: {prompt}\nA:", 
        max_tokens=1024, n=1, stop=None, temperature=0.7,
    )
    answer = response.choices[0].text.strip()
    
    # Return answer as JSON
    #return jsonify({"answer": answer})
    print(answer)
    return render_template("answer.html", response=answer) 


# if __name__ == "__main__":
#     app.run() 

