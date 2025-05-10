# import os
# from dotenv import load_dotenv, find_dotenv
# _ = load_dotenv(find_dotenv())

# # Replace this with your Gemini API Key
# GOOGLE_API_KEY = os.environ["GOOGLE_API_KEY"]

# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# from langchain_chroma import Chroma
# from langchain_community.document_loaders import TextLoader
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.prompts import HumanMessagePromptTemplate
# from langchain_core.prompts import PromptTemplate

# # Load the model (Gemini Pro)
# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=GOOGLE_API_KEY)

# # Load and split documents
# loader = TextLoader("./data/be-good.txt")
# docs = loader.load()

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# # Use Gemini-compatible embeddings
# vectorstore = Chroma.from_documents(documents=splits, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY))

# retriever = vectorstore.as_retriever()

# # Prompt
# prompt  = ChatPromptTemplate(
#     input_variables=['context', 'question'],
#     metadata={
#         'lc_hub_owner': 'rlm',
#         'lc_hub_repo': 'rag-prompt',
#         'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'
#     },
#     messages=[
#         HumanMessagePromptTemplate(
#             prompt=PromptTemplate(
#                 input_variables=['context', 'question'],
#                 template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
#             )
#         )
#     ]
# )

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# # Run the chain
# response = rag_chain.invoke("What is this article about?")

# print("\n----------\n")
# print("What is this article about?")
# print("\n----------\n")
# print(response)
# print("\n----------\n")


import os

from dotenv import load_dotenv, find_dotenv

_= load_dotenv(find_dotenv())

google_api_key = os.environ["GOOGLE_API_KEY"]

from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI

llm = GoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=google_api_key,temperature=0.1)

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

loader = TextLoader('data/be-good.txt')
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

splits = text_splitter.split_documents(docs)

vector_store = Chroma.from_documents(documents=splits,embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=google_api_key
))

retriever = vector_store.as_retriever()

# Prompt
prompt  = ChatPromptTemplate(
    input_variables=['context', 'question'],
    metadata={
        'lc_hub_owner': 'rlm',
        'lc_hub_repo': 'rag-prompt',
        'lc_hub_commit_hash': '50442af133e61576e74536c6556cefe1fac147cad032f4377b60c436e6cdcb6e'
    },
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'],
                template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"
            )
        )
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is this article about")

print("\n----------\n")
print("What is this article about?")
print("\n----------\n")
print(response)
print("\n----------\n")