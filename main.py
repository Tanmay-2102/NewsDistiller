import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader, UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from transformers import pipeline

HUGGINGFACEHUB_API_TOKEN = st.secrets["HUGGINGFACEHUB_API_TOKEN"]

st.title("📈 NewsDistiller: News Research Tool")
st.sidebar.title("News Source Configuration")

loader_choice = st.sidebar.selectbox(
    "Choose content loader",
    ["WebBaseLoader (Fast)", "UnstructuredURLLoader (Better Parsing)"]
)

urls = [st.sidebar.text_input(f"URL {i+1}") for i in range(3)]
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_store.pkl"
main_placeholder = st.empty()

hf_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

llm = HuggingFacePipeline(pipeline=hf_pipeline)

if process_url_clicked:
    with st.spinner("Loading and processing URLs..."):
        data = []

        non_empty_urls = [url.strip() for url in urls if url.strip()]
        if not non_empty_urls:
            st.error("❌ Please enter at least one valid URL.")
            st.stop()

        try:
            if loader_choice.startswith("WebBase"):
                for url in non_empty_urls:
                    loader = WebBaseLoader(url)
                    data.extend(loader.load())
            else:
                loader = UnstructuredURLLoader(urls=non_empty_urls)
                data.extend(loader.load())
        except Exception as e:
            st.error(f"Failed to load content: {e}")
            st.stop()

        if not data:
            st.error("❌ No content extracted from the URLs.")
            st.stop()

        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)

        if not docs:
            st.error("❌ No text chunks were created.")
            st.stop()

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        embeddings = HuggingFaceEmbeddings(model_name=model_name)

        vectorstore = FAISS.from_documents(docs, embeddings)
        with open(file_path, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Articles processed and vectorstore saved!")

query = st.text_input("Ask a question about the articles:")
if query:
    if not os.path.exists(file_path):
        st.warning("Please process some URLs first.")
    else:
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)

        with st.spinner("Thinking..."):
            # Retrieve the relevant documents using the retriever
            docs = retriever.get_relevant_documents(query)
            
            # Prepare the context for the question-answering model
            context = " ".join([doc.page_content for doc in docs])  # Corrected here
            
            # Format the input for the HuggingFace pipeline correctly
            hf_input = {
                'context': context,
                'question': query
            }
            
            # Pass the formatted input to the pipeline
            result = hf_pipeline(hf_input)

        st.header("Answer")
        st.write(result.get("answer", "No answer found."))

        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources")
            for src in sources.split(","):
                st.write(f"- {src.strip()}")
        else:
            st.info("No sources returned.")
