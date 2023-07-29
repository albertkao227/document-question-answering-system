#!/usr/bin/python3

import os
import numpy as np
import openai
import pickle
import requests
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)
openai.api_key = os.environ.get("OPENAI_API_KEY")


def read_pickle(filename):
    with open(filename, 'rb') as f:
        content = pickle.load(f)
    return content


def get_top3(qvector, pvectors, n):
    dot_prod = np.sum(qvector*pvectors, axis=1)
    pnorm2 = np.sum(pvectors**2, axis=1)
    dist = dot_prod/pnorm2
    results = dist.argsort()[-n:]
    return results
    

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/answer", methods=["POST"])
def answer():
    question = request.form["question"]
    category = request.form["category"]
    
    page_vectors = read_pickle('vectors.pkl') 
    page_text = read_pickle('manual.pkl') 
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    question_vec = model.encode([question])[0]
     
    top3 = get_top3(question_vec, page_vectors, 3)
    print(top3)
    print(page_text[top3[0]])   
    print(page_text[top3[1]])  
    print(page_text[top3[2]])  


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


# Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES").
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.
# ALWAYS return a "SOURCES" part in your answer.

# QUESTION: {question}
# =========
# Content: ...
# Source: ...
# ...
# =========
# FINAL ANSWER:
# SOURCES: