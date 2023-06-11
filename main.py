from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
# import os
import nltk
# YOur api key here
openai_api_key = "Yourapikey"


loader = DirectoryLoader(r"LangChainTrain", glob='**/*.txt')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5)

texts = text_splitter.split_documents(documents)
# print(texts)
# turn texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# searching over document
docsearch = FAISS.from_documents(texts, embeddings)
# load llm
llm = OpenAI(openai_api_key=openai_api_key)
# create retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = docsearch.as_retriever())
# run query
query = "What if i want to cancel  my order?"
qa.run(query)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk
openai_api_key = "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"


loader = DirectoryLoader(r"C:\Users\xmate\Desktop\LangchainTrain", glob='**/*.txt')

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=5)

texts = text_splitter.split_documents(documents)
# print(texts)
# turn texts into embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# searching over document
docsearch = FAISS.from_documents(texts, embeddings)
# load llm
llm = OpenAI(openai_api_key=openai_api_key)
# create retriever
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever = docsearch.as_retriever())
# run query
while True:

    query = input("How can i help you ?")
    print(qa.run(query))
