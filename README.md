# document-question-answering-system

## Utilize large language models to develop QA system for provided documents.

This repository utlizes LLM, or [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/text-generation/json-mode) to create an app for question-answering for the contents in documents.     


Download python libraries:

```
pip install -r requirements.txt
```

To vectorize documents for the question-answering system using [sentence transformers](https://sbert.net/) with example instructions [here](https://sbert.net/#usage).

Convert each section or each page of the target document into one single vector and save. 

To run the app, move to top directory and type: 

```
flask run
```
we can see an application opening up in your browser!

![User Interface](assets/example.png)



