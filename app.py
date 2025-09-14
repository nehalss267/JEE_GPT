!pip install gradio langchain langchain_community langchain_milvus transformers pymilvus
!pip install sentence-transformers
!pip install google-generativeai langchain-google-genai

import os
import gradio as gr
import tempfile
from langchain_milvus import Milvus
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # ✅ Gemini LangChain wrapper

# Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBmYjE8gzaHozjIdwuGOGxLFEtzEUw7o6g"

# Define Gemini model (LangChain wrapper)
gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

# DB setup
db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)

# Load nutrition knowledge base
filename = "StudyGPT_knowledgeBase.pdf"
# with open(filename, "w") as f:
#     f.write("""
# """)
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(filename)
documents = loader.load()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=tokenizer.model_max_length // 2,
    chunk_overlap=0,
)
texts = splitter.split_documents(documents)
for i, doc in enumerate(texts):
    doc.metadata["doc_id"] = i + 1
vector_db.add_documents(texts)

# Prompt
template = """
You are StudyGPT, an AI tutor specializing in clearing JEE (Joint Entrance Examination) doubts.
User Question: {question}
Instructions for your answer:
1. Clearly identify the subject (Physics / Chemistry / Mathematics).
2. Break the solution into step-by-step reasoning.
3. Use formulas, derivations, and standard notations where required.
4. Provide the final answer at the end.
5. Keep explanations simple, precise, and exam-focused.
6. If multiple methods exist, show the easiest and most efficient JEE approach.
7. Add quick tips or shortcuts if applicable.
Always explain in a way that builds conceptual clarity and helps the student prepare better for JEE.
"""

prompt = PromptTemplate(template=template, input_variables=["question"])

# ✅ Gemini LLM works directly here now
llm_chain = LLMChain(llm=gemini_llm, prompt=prompt)
combine_chain = StuffDocumentsChain(
    llm_chain=llm_chain,
    document_variable_name="question"
)

rag_chain = RetrievalQA(
    retriever=vector_db.as_retriever(),
    combine_documents_chain=combine_chain,
    return_source_documents=False
)

def ask_StudyGPT(query):
    try:
        response = rag_chain.run(query)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

iface = gr.Interface(
    fn=ask_StudyGPT,
    inputs=gr.Textbox(
        label="Ask a question",
        placeholder="e.g. What is centripetal force"
    ),
    outputs=gr.Textbox(label="Answer"),
    title="StudyGPT",
    description="Ask Physics,Chemistry,Maths doubts",
    theme="default"
)

iface.launch()
