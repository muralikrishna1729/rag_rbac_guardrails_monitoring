from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

load_dotenv()

def build_rag_chain(persist_directory: str):
    text_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory= persist_directory ,embedding_function= text_embedding)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})   

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature = 0.1 ,# low temp = more factual

    )

    prompt = ChatPromptTemplate.from_template(
                """
                You are a helpful assistant for company employees.
                Use only the context below to answer the question.
                If the answer is not in the context, say "I don't have that information."
                Context: {context}
                Question: {question}
                Answer:
                """
    )

    chain = ({"context": retriever | format_docs ,"question":RunnablePassthrough()}| prompt | ChatHuggingFace(llm=llm) | StrOutputParser())
    return chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__== "__main__":
    chain = build_rag_chain("./chroma_db")
    response = chain.invoke("What is the leave policy?")
    print(response)




