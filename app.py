from flask import Flask, render_template, jsonify, request
from src.helper import downlaod
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os
import cohere

app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

embeddings=downlaod()

docsearch =  PineconeVectorStore.from_existing_index(
    index_name="test",
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type='similarity', search_kwargs ={"k":3})

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)


co = cohere.Client('F4rq95B30Awcrlr51d1A3WSwdJwRAUIoe5ggmDed')

# Function to generate a response using Cohere
def cohere_generate(context, question):
    prompt = f"Answer the following question based on the context below: \n\nContext: {context} \n\nQuestion: {question}"
    response = co.generate(
        model='command-r-plus',  # Choose the model size
        prompt=prompt,
        max_tokens=600,
        temperature=0.7
    )
    return response.generations[0].text.strip()

# Set up Pinecone VectorStore for document retrieval
docsearch = PineconeVectorStore.from_existing_index(
    index_name="test",
    embedding=embeddings  # Assumed to be defined earlier
)

# Set up the retriever with similarity search
retriever = docsearch.as_retriever(search_type='similarity', search_kwargs={"k": 3})

# Define the system prompt
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answers concise."
    "\n\n"
    "{context}"
)

# Create the ChatPromptTemplate for Cohere
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# Custom QA chain for Cohere model (Cohere QA Chain)
class CohereQAChain:
    def __init__(self, prompt_template):
        self.prompt_template = prompt_template

    def run(self, context, question):
        # Format the prompt with context and question
        formatted_prompt = self.prompt_template.format(context=context, input=question)
        
        # Use Cohere to generate the answer based on the context and question
        answer = cohere_generate(context, question)
        return answer

# Create the Retrieval-Augmented Generation (RAG) chain
def create_retrieval_chain_with_rag(retriever, question_answering_chain):
    def chain(question):
        # Retrieve relevant documents using the retriever
        docs = retriever.get_relevant_documents(question)
        context = "\n".join([doc.page_content for doc in docs])
        
        # Use the question-answering chain to generate the answer
        answer = question_answering_chain.run(context, question)
        return answer
    return chain

# Initialize the QA chain with Cohere model
question_answering_chain = CohereQAChain(prompt)

# Set up the RAG chain
rag_chain = create_retrieval_chain_with_rag(retriever, question_answering_chain)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/get', methods=["GET","POST"])
def chat():
    msg = request.form.get("msg")
    if not msg:
        return jsonify({"error": "Message is required"}), 400

    answer = rag_chain(msg)  # Process the message with your model
    return jsonify({"answer": answer}) 


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)

