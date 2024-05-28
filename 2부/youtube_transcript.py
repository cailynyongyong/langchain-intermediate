from langchain_community.document_loaders.blob_loaders.youtube_audio import (
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")

# set a flag to switch between local and remote parsing
# change this to True if you want to use local parsing
local = False

# Two Karpathy lecture videos
urls = ["https://youtu.be/fcfQkxwz4Oo?si=0er99KSrEOMrt9HE"]

# Directory to save audio files
save_dir = "~/Downloads/YouTube"

# Transcribe the videos to text

loader = GenericLoader(YoutubeAudioLoader(urls, save_dir), OpenAIWhisperParser())
docs = loader.load()

from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Combine doc
combined_docs = [doc.page_content for doc in docs]
text = " ".join(combined_docs)

# Split them
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_text(text)

# Build an index
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_texts(splits, embeddings)

# Build a QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=vectordb.as_retriever(),
)

# Ask a question!
query = "Brief summarization on quantum mechanics?"
result = qa_chain.run(query)
print(result)