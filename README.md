# AI Chatbot Backend Workshop

This repository contains materials for the **AI Chatbot Backend Development (Hands-on) Workshop** happened on 3.04.2026 at Tashkent International University of Education.

During the workshop, we built a simple AI chatbot backend step by step using a Large Language Model (LLM) and basic Retrieval-Augmented Generation (RAG).

---

## 📌 Workshop objectives

- How AI chatbots work (high-level architecture)
- How to send requests to an LLM (Gemini)
- How to maintain conversation history
- Basics of Retrieval-Augmented Generation (RAG)
- Simple safety mechanisms for user input
- How backend connects to any frontend (web, app, etc.)

---

## 🧠 Project Structure

```
.
├── simple_chatbot_answers.ipynb # simple chatbot we went through during the workshop
├── chatbot_backend_exercise.ipynb # main code
├── app.py # backend file
├── app_py_explained.ipynb # a step-by-step explanation of app.py in a jupyter file
├── templates/index.html # frontend file
├── utilities/
    ├── crawler.py # used to crawl TIUE en webpages
    ├── loader.py # used to load the documents into Pinecone database
    ├── requirements.txt # need to be installed via pip within a virtual environment before running app.py
    ├── tiue_en_pages.json # document that has been loaded to the Pinecone database
    ├── utilities.py # contains practically better example prompt
    ├── RAG_notebook.ipynb # inspiration notebook that I found at https://github.com/AlaGrine/RAG_chatabot_with_Langchain
├── .env # keep your api keys here
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository
```
git clone <your-repo-link>
cd <repo-name>
```

### 2. Create virtual environment (recommended)
```
python -m venv .venv
source .venv/bin/activate
.venv\Scripts\activate
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Put into `.env` file the API you got from Google AI Studio:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## 🚀 Running the Backend
Once you set up things above with no errors, run the following command from the same directory as the app.py:

```
flask run
```

Open the following link in your browser:
```
http://localhost:5000/chat
```

---

## 💻 How It Works

```
User Input → LLM → Response
```

With memory:
```
User Input + Chat History → LLM → Response
```

With RAG:
```
User Input → Retrieve Data → LLM → Response
```

---

## 🛡️ Safety

- Detect prompt injection attempts
- Limit input length
- Prevent unsafe instructions

---

## 👩‍💻 Workshop Goal

By the end:
- Working chatbot backend
- Understanding of AI systems
- Portfolio-ready project

---

Happy coding and feel free to explore the repo! 🚀

