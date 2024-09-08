import streamlit as st
from pinecone import Pinecone
import cohere
import numpy as np
import time
from src.llms import gemini,groq

st.title("🔮 बोल  भिडू !")
st.subheader(" ",divider='rainbow')

co = cohere.Client(st.secrets.COHERE_API_KEY)
pc = Pinecone(api_key=st.secrets.PINECONE_API_KEY)

index = pc.Index("marathi-rag")
chunks = np.load('chunks.npy')

def query_embd(query):
    response = co.embed(
        texts=[query],
        model="embed-multilingual-v3.0",
        input_type="search_query",
        embedding_types=['float']
    )
    query_embedding = response.embeddings.float[0]
    return query_embedding



def matched_ids(query_embedding):

    query_results = index.query(
        # namespace="example-namespace1",
        vector=query_embedding,
        top_k=3,
        include_values=True
    )
    # query_results1

    ids = [int(match['id']) for match in query_results['matches']]

    return ids


# YouTube video URL
query = st.text_input("Enter Your query")

submit = st.button("submit")

if submit:
    start_time = time.time()

    query_embedding = query_embd(query)
    id_list = matched_ids(query_embedding)
    st.write(id_list)
    context = "\n\n".join(chunks[i] for i in id_list)
    st.write(context)

    st.write('---')
    res = gemini(context,query)
    st.write(res)
    st.write('---')

    st.write('---')
    res = groq(context,query,"gemma-7b-it")
    st.write(res)
    st.write('---')

    end_time = time.time()

    st.write(f"Time taken to run: {end_time - start_time:.2f} seconds")







