from langchain.llms import GPT4All
from langchain.vectorstores import Chroma
from langchain.embeddings import GPT4AllEmbeddings
from langchain.document_loaders import WebBaseLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


def initalize_llm():
    llm = GPT4All(
    model="D:/Program Files (x86)/gpt4all/models/ggml-model-gpt4all-falcon-q4_0.bin",
    max_tokens=2048
    )
    return llm
    

def initialize_vector_database():
    embeddings = GPT4AllEmbeddings()
    vectorstore = Chroma("langchain_store", embeddings)
    return vectorstore

def add_website(url, db):
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    db.add_documents(all_splits)

def add_pdf(doc, db):
    loader = WebBaseLoader()
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(data)

    db.add_documents(all_splits)

def pass_prompt(llm, query, db):
    print(query)
    # Prompt
    prompt = PromptTemplate.from_template(
        "Summarize the main themes in these retrieved docs: {docs}"
    )

    # Chain
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Run
    docs = db.similarity_search(query)
    result = llm_chain(docs)
    print(result["text"])
    return result["text"]