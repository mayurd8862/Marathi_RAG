import getpass
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import streamlit as st


model =["llama3-8b-8192","gemma-7b-it","llama3.1-8b-instant"]

def groq(context,query, model):

    template = f"""Use the following pieces of context to answer the user question. This context retrieved from a knowledge base and you should use only the facts from the context to answer.
    Your answer must be based on the context. If the context not contain the answer, just say that 'I don't know', don't try to make up an answer, use the context.
    Don't address the context directly, but use it to answer the user question like it's your own knowledge.
    Use three sentences maximum. answer should be in Marathi.

    Context:
    {context}

    Question: {query}
    """

    chat = ChatGroq(temperature=0, groq_api_key=st.secrets.GROQ_API_KEY, model_name=model)
    system = "You are a helpful assistant."
    human = "{text}"
    prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

    chain = prompt | chat
    result = chain.invoke({"text": template})
    return result.content


def gemini(context,query):
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass(st.secrets.GOOGLE_API_KEY)

    template = f"""Use the following pieces of context to answer the user question. This context retrieved from a knowledge base and you should use only the facts from the context to answer.
    Your answer must be based on the context. If the context not contain the answer, just say that 'I don't know', don't try to make up an answer, use the context.
    Don't address the context directly, but use it to answer the user question like it's your own knowledge.
    Use three sentences maximum. answer should be in Marathi.

    Context:
    {context}

    Question: {query}
    """

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    result = llm.invoke(template)
    return result.content
    