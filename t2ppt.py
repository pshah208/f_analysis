import os
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import sqlite3
import base64
import openai
import pptx
from pptx.util import Inches, Pt
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader, UnstructuredPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

# Get an OpenAI API Key before continuing
if "openai_api_key" in st.secrets:
    openai_api_key = st.secrets.openai_api_key
else:
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Enter an OpenAI API Key to continue")
    st.stop()
llm = ChatOpenAI(openai_api_key=openai_api_key)
load_dotenv(find_dotenv())
embeddings= OpenAIEmbeddings(openai_api_key=openai_api_key)

#User input query
ppt_query= st.text_input('Please enter your topic ')

#creating a database
def creating_db(ppt_query):
    
 loader = DirectoryLoader('./docs/', glob="**/[!.]*", loader_cls=UnstructuredPDFLoader)
 docs = loader.load()


 splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=20)

 documents = splitter.split_documents(docs)
 documents[0]
# Create vector embeddings and store them in a vector database
 db = Chroma.from_documents(documents, embedding=OpenAIEmbeddings())    
    
 return db


