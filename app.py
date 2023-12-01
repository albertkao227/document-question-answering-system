#!/usr/bin/python3

import os
import json
import numpy as np
import openai
import pickle
import requests
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from flask import Flask, redirect, render_template, request, url_for
model = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)
api_key = os.environ.get("OPENAI_API_KEY")


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
    page_vectors = read_pickle('vectors1.pkl') 
    page_text = read_pickle('manual.pkl') 

    question_vec = model.encode([question])[0]     
    top3 = get_top3(question_vec, page_vectors, 3)
    print(top3)
    print(page_text[top3[0]])   
    print(page_text[top3[1]])  
    print(page_text[top3[2]])  
    document = page_text[top3[0]]

    if category=='Summarize':
        prompt = f'''
        Please summarize below paragraph into three sentences:
        {question}                                                                                                           
        Answer: 
        '''
    elif category=='Question Answering':
        prompt = f'''
        Please answer below question to the best of your ability and provide reference. 
        QUESTION: {question}        
        =========                                                                                                   
        ANSWER: 
        '''        
    else:
        # prompt = f'''
        # Please answer following question in details:
        # {question}                                                                                                           
        # Answer: 
        # '''

        prompt = f'''
        Please respond as if you were talking to a non-expert. Use analogies. 
        Given the following extracted parts of a operating manual of a space shuttle and a question, 
        create a final answer with references from the provided document.
        If you don't know the answer, just say that you don't know. 
        Don't try to make up an answer.
        ALWAYS return a "SOURCES" part in your answer.
        DOCUMENT: {document}
        QUESTION: {question}
        ...
        =========
        FINAL ANSWER:
        SOURCES:
        '''

    client = OpenAI(
        api_key=f"{api_key}",
    )
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": f"{prompt}\n",
            }
        ],
        #model="gpt-3.5-turbo",
        model='gpt-4',
    )

    response = json.loads(completion.model_dump_json())
    answer = response['choices'][0]['message']['content']
    output = f'{category}: \\n page: {top3[0]} \\n {answer}'
    return render_template("answer.html", response=output) 


# if __name__ == "__main__":
#     app.run() 
