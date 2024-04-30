import argparse
import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")


def read_document(filepath):
    with open(filepath) as f:
        lines = f.read().splitlines()
    return lines 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vectorize text")
    parser.add_argument("--document", type=str, help="Document to vectorize")

    args = parser.parse_args()
    filepath = args['document']  
    content = read_document(filepath) 
    embeddings = model.encode(content)

    with open('vectors.pkl', 'wb') as file:
        # Serialize and write the variable to the file
        pickle.dump(embeddings, file)

