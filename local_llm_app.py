import os
import pickle  
from prompts import * 
from llama_cpp import Llama
from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)
model_path = os.environ.get("MODEL_PATH")


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


llm = Llama(
    model_path=model_path,
    n_ctx=3000,  # Context length to use
    n_gpu_layers=30,  # Number of layers to use on GPU 
)


generation_kwargs = {
    "max_tokens":20000,
    "stop":["</s>"],
    "echo":False, # Echo the prompt in the output
    "top_k":10 
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/answer", methods=["POST"])
def answer():
    question = request.form["question"]
    category = request.form["category"]
    page_vectors = read_pickle('vectors.pkl') 
    page_text = read_pickle('document.pkl') 

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

    res = llm(prompt, **generation_kwargs) 
    answer = res["choices"][0]["text"]
    if category == 'Answer Questions about Document':
        output = f'{category}: | Page: {top3[0]} | Answer: {answer}'
    else:
        output = f'{category}: | Answer: {answer}'
    return render_template("answer.html", response=output) 


