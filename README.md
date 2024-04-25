# document-question-answering-system

## Utilize large language models to develop QA system for provided documents.

This repository utlizes LLM, or [OpenAI ChatGPT API](https://platform.openai.com/docs/guides/text-generation/json-mode) to create an app for question-answering for the contents in documents.     


Download python libraries:

```
pip install -r requirements.txt
```

### Preprocess Documents for Similarity Search 

To vectorize documents for the question-answering system, use library [sentence transformers](https://sbert.net/) with example instructions [here](https://sbert.net/#usage). Convert each section of the document into a single vector, and save to `vectors.pkl`. The original document should be saved as `document.pkl`. 


### Run App 

To run the app: 

```
flask run
```

QA Application should open up in browser: 

![User Interface](assets/example.png)



