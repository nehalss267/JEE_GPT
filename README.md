```markdown
# 📘 StudyGPT – JEE Doubt Clearing Assistant

StudyGPT is an AI-powered **JEE (Joint Entrance Examination) tutor** that clears student doubts in **Physics, Chemistry, and Mathematics**.  
It uses:
- **Google Gemini (Generative AI)** as the LLM
- **LangChain** for RAG (Retrieval-Augmented Generation)
- **Milvus** as the vector database
- **HuggingFace embeddings**
- **Gradio** for the web interface
- A custom **knowledge base (PDF/TXT)** with study material

---

## 🚀 Features
- Upload your **knowledge base** (PDF or TXT).
- Asks and answers **JEE-level doubts** with step-by-step solutions.
- Uses **RAG pipeline** for factual answers.
- Provides **exam tips, shortcuts, and conceptual clarity**.
- Simple **Gradio UI** for interaction.

---

## 📂 Project Structure
```

StudyGPT/
│-- StudyGPT\_knowledgeBase.pdf   # Your study material (can be PDF/TXT)
│-- app.py                       # Main Python app
│-- requirements.txt             # Dependencies
│-- README.md                    # This file

````

---

## ⚡ Setup Instructions

### 1️⃣ Clone the repo & move inside
```bash
git clone <your-repo-url>
cd StudyGPT
````

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```
gradio
langchain
langchain-community
langchain-milvus
langchain-google-genai
transformers
pymilvus
sentence-transformers
pypdf
google-generativeai
```

### 3️⃣ Get Gemini API Key

* Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
* Click **Create API Key**
* Copy it and set in your environment:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

### 4️⃣ Run the app

```bash
python app.py
```

---

## 🧠 Usage

1. Upload your **study material** in `StudyGPT_knowledgeBase.pdf`.
2. Ask any **JEE question** in Physics, Chemistry, or Math.
3. Get a **step-by-step explanation + final answer**.
4. Use it for **concept clarification + practice**.

---

## 🎯 Example

**Question:**

> Find the work done in moving a charge of 2C through a potential difference of 12V.

**Answer (StudyGPT):**

* Subject: Physics (Electrostatics)
* Step 1: Formula → Work = q × V
* Step 2: Substitute → 2 × 12 = 24 J
* ✅ Final Answer: **24 Joules**
* Tip: Always remember Work = Charge × Potential difference.

---

## 🔮 Future Scope

* Add support for **multiple PDFs** as knowledge base.
* Generate **practice tests** automatically.
* Add **voice-based doubt clearing**.

---

## 🤝 Contributing

PRs are welcome! Please fork the repo, create a branch, and submit a pull request.

---
