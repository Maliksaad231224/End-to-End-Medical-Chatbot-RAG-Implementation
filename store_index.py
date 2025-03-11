from src.helper import load_pdf, text_split, downlaod
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

extracted = load_pdf(data='/home/malik-saad-ahmed/Desktop/Learnings/Medical Chatbot RAG/RAG-Medical-Chatbot-/data/')
text_chunks= text_split(extracted)
embeddings = downlaod()

pc = Pinecone(api_key=PINECONE_API_KEY)

index_name= 'test'

pc.create_index(
    name=index_name,
    dimension=384,
    metric='cosine',
    spec=ServerlessSpec(
        cloud='aws',
        region='us-east-1'
    )
)

docsearch =  PineconeVectorStore.from_existing_index(
    index_name="test",
    embedding=embeddings,
    documents = text_chunks
)

