import streamlit as st
import os
import time
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

# Ensure an asyncio loop exists (Fix for Streamlit & Torch issue)
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Gemini API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")        # Groq API Key

# Streamlit UI Title
st.title(" AI-Powered Document Query 'AskMyDocs'" )

# Initialize LLM (Groq for chatbot)
llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model_name="Llama3-8b-8192")

# Define Prompt
prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context:
    <context>
    {context}
    </context>
    Question: {input}
    Answer:"""
)

# Initialize session state variables if they don't exist
for key in ["docs", "final_docs", "embeddings", "vectors"]:
    if key not in st.session_state:
        st.session_state[key] = None

def load_pdfs():
    """Loads PDFs, splits them into chunks, and prepares for embedding."""
    loader = PyPDFDirectoryLoader("./wef")
    st.session_state.docs = loader.load()

    if not st.session_state.docs:
        st.error(" No PDFs found! Check the folder path or ensure PDFs exist.")
        return

    # Split the documents
    text_splitter = RecursiveCharacterTextSplitter()
    st.session_state.final_docs = text_splitter.split_documents(st.session_state.docs)

    if st.session_state.final_docs:
        st.success(f" Loaded {len(st.session_state.final_docs)} document chunks!")
    else:
        st.error(" Document splitting failed. Check the PDF contents.")

def vector_embedding():
    """Creates vector embeddings using Gemini embeddings."""
    if st.session_state.final_docs is None or not st.session_state.final_docs:
        st.error(" No split documents found. Load PDFs first.")
        return

    if st.session_state.embeddings is None:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS index using the Gemini embeddings
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
    st.success(" Vector embeddings successfully created using Gemini!")

st.write(" Session state:", st.session_state)

# Input Field for Questions
prmpt1 = st.text_input(" Ask a question from the documents:")

# Load PDFs Button
if st.button(" Load PDFs"):
    load_pdfs()
    st.write(" Documents Loaded!")

# Create Embeddings Button
if st.button(" Create Embeddings"):
    vector_embedding()
    st.write(" Vector store database created!")

# Process User Query
if prmpt1:
    start = time.process_time()

    if st.session_state.vectors is None:
        st.error(" Embeddings have not been created yet. Click 'Create Embeddings' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retriever_chain = create_retrieval_chain(retriever, document_chain)
        response = retriever_chain.invoke({"input": prmpt1})

        # Safely retrieve response content
        answer = response.get("answer", "No answer received.")
        context_docs = response.get("context", [])

        st.write(answer)

        # Display document similarity search results
        with st.expander("🔎 Document Similarity Search"):
            if context_docs:
                for i, doc in enumerate(context_docs):
                    st.write(doc.page_content)
                    st.write("────────────────────")
            else:
                st.write("No similar documents found.")

    end = time.process_time()
    st.write(f" Response time: {end - start:.2f} seconds")
