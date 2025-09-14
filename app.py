pip install gradio langchain langchain_community langchain_milvus transformers pymilvus
pip install sentence-transformers
pip install google-generativeai langchain-google-genai
pip install pypdf
pip install pdfplumber
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.document_loaders import PyPDFLoader
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
from langchain_google_genai import ChatGoogleGenerativeAI  

#Steps to create Gemini API Key in README.md
os.environ["GOOGLE_API_KEY"] = "<API_KEY>"

gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.environ["GOOGLE_API_KEY"])

db_file = tempfile.NamedTemporaryFile(prefix="milvus_", suffix=".db", delete=False).name
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": db_file},
    auto_id=True,
    index_params={"index_type": "AUTOINDEX"},
)
from langchain.text_splitter import CharacterTextSplitter
from transformers import AutoTokenizer

from langchain_community.document_loaders import PDFPlumberLoader
filename = "StudyGPT_knowledgeBase.pdf"
loader = PDFPlumberLoader(filename)
documents = loader.load()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

splitter = CharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=500,
    chunk_overlap=50
)

docs = splitter.split_documents(documents)

template = """You are StudyGPT, an AI tutor specializing in clearing JEE (Joint Entrance Examination) doubts.User Question: {question}"""

prompt = PromptTemplate(template=template, input_variables=["question"])

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
