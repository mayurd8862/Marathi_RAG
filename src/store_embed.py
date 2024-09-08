
from langchain_text_splitters import RecursiveCharacterTextSplitter
import cohere
import streamlit as st
import numpy as np
import os
from pinecone import Pinecone, ServerlessSpec

co = cohere.Client(st.secrets.COHERE_API_KEY)
pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)


with open("C:/Users/mayur dabade/Desktop/Projects/marathi RAG/data/maharaj.txt", encoding='utf-8') as f:
    text = f.read()

print(f"The text has roughly {len(text.split())} words.")


# Create basic configurations to chunk the text
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    is_separator_regex=False,
)

# Split the text into chunks with some overlap
chunks_ = text_splitter.create_documents([text])
chunks = [c.page_content for c in chunks_]
print(f"The text has been broken down in {len(chunks)} chunks.")

# Create a NumPy array
arr = np.array(chunks)

# Save the array to a .npy file
np.save('chunks.npy', arr)

# Because the texts being embedded are the chunks we are searching over, we set the input type as search_doc
model="embed-multilingual-v3.0"
response = co.embed(
    texts= chunks,
    model=model,
    input_type="search_document",
    embedding_types=['float']
)
embeddings = response.embeddings.float
print(f"We just computed {len(embeddings)} embeddings.")

# Create a serverless index
pc.create_index(name="marathi-rag", dimension=len(embeddings[0]), 
                spec=ServerlessSpec(cloud='aws', region='us-east-1'))

# Target the index
index = pc.Index("marathi-rag")

# Prepare the vectors for upsert (only id and embedding)
vectors_to_upsert = []
for i, embedding in enumerate(embeddings):
    vector = {
        "id": f"{i}",
        "values": embedding
    }
    vectors_to_upsert.append(vector)

# Upsert vectors into the Pinecone index
index.upsert(vectors=vectors_to_upsert)

print("Vectors successfully upserted into Pinecone!")