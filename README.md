# Gemini RAG Application 🔍🧠

This is a simple Retrieval-Augmented Generation (RAG) application powered by **Google's Gemini (via LangChain)**. It loads a local `.txt` file, splits the content into chunks, generates embeddings using Gemini, stores them in a Chroma vector database, and answers user queries using a Gemini LLM.

---

## 🚀 Features

- 🔎 Loads and processes a `.txt` document (`data/be-good.txt`)
- 🧩 Splits content into chunks using `RecursiveCharacterTextSplitter`
- 🧠 Embeds using `GoogleGenerativeAIEmbeddings`
- 🗃️ Stores vectors using ChromaDB
- 💬 Answers questions using Gemini Pro (`gemini-2.0-flash`)
- 🔄 RAG chain with a custom prompt template

---

## 📂 Project Structure

├── data/
│ └── be-good.txt # Your source document
├── .env # Contains your GOOGLE_API_KEY
├── app.py # Main application script


---

## 🧪 Dependencies

Make sure you install the following dependencies:

    pip install langchain langchain-google-genai langchain-community langchain-core chromadb python-dotenv

## 🔐 Environment Variables
Create a .env file in the root directory and add your Gemini API key:


GOOGLE_API_KEY=your_gemini_api_key
## 📜 Usage
Run the Python script to ask a question based on the content of the be-good.txt file:

python app.py

Output:
----------
What is this article about?
----------
<response from Gemini>
----------
  
## 🧠 How It Works (Code Walkthrough)
Environment Setup
Loads Gemini API key from .env.

Document Loading & Splitting

Loads a .txt file using TextLoader.

Splits the text into 1000-character chunks with 200-character overlap.

Embeddings & Vector Storage

Generates embeddings using Gemini (embedding-001).

Stores them in a Chroma vector store.

Prompt Template
Uses a custom concise Q&A format.

RAG Chain Execution
Retrieves relevant context and invokes Gemini LLM to answer the query.

## 💡 Example Prompt Template
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Question: {question}  
Context: {context}  
Answer:

## 🤝 Credits
LangChain

Google Generative AI

Chroma Vector DB

## 📬 License
This project is open source and free to use under the MIT License.
