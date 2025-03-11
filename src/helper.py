from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(data):
    loader=DirectoryLoader(data,
                           glob="*.pdf", 
                           loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

def text_split(extracted_Data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=20)
    text_chunk = text_splitter.split_documents(extracted_Data)
    return text_chunk

def downlaod():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings

