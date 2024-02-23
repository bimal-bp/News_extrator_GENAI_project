import os 
import streamlit as st 
import pickle 
import time 
from langchain import OpenAI 
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS 

from dotenv import load_dotenv
load_dotenv()

st.title("ORBT bot : News Research Tool :")
st.sidebar.title("News article URLS  ")

urls= []
for i in range(3):
    url=st.sidebar.text_input(f"URL{i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLS")

file_path="faiss_store_openai.pkl"

main_placeholder=st.empty()
llm=OpenAI(temperature=0.8,max_tokens=500)


if process_url_clicked:
    # data load
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text(" Data Loading is on:")
    data=loader.load()

    # spliting data
    text_splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n','.',','],
        chunk_size=1000
    )
    main_placeholder.text(" Text splitter start ::")

    docs=text_splitter.split_documents(data)

    # embedding 

    embedding=OpenAIEmbeddings()
    vectorstore_openai=FAISS.from_documents(docs,embedding)
    main_placeholder.text("Embedding is working ::")

    time.sleep(2)

    with open(file_path,"wb") as f:
        pickle.dump(vectorstore_openai,f)

    
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path,"rb") as f:
            vectorstore=pickle.load(f)
            chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
            result=chain({"question:query"},return_only_outputs=True)
            st.header("Answer")
            st.write(result["answer"])


            source=result.get("source","")

            if source:
                st.subheader("Source:")
                source_list=source.split("\n")

                for source in source_list:
                    st.write(source)



