from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from app.auth.users import get_user_role

load_dotenv()
text_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_rag_chain(persist_directory: str, role:str):
    vectorstore = Chroma(persist_directory= persist_directory ,embedding_function= text_embedding)

    search_kwargs = {"k":3}
    if role != 'admin':
        search_kwargs["filter"] = {"$or" : [{"role":role},{"role":"general"}]}
    else:
        print("Admin access granted: Searching all departments.")

    retriever = vectorstore.as_retriever(
            search_kwargs=search_kwargs     
        )   
    # llm = HuggingFaceEndpoint(
    #     repo_id="meta-llama/Llama-3.1-8B-Instruct",
    #     task="text-generation",
    #     temperature = 0.1 ,# low temp = more factual

    # )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
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

    # chain = ({"context": retriever | format_docs ,"question":RunnablePassthrough()}| prompt | ChatHuggingFace(llm=llm) | StrOutputParser())
    chain = ({"context": retriever | format_docs ,"question":RunnablePassthrough()}| prompt | llm | StrOutputParser())

    return chain

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

if __name__== "__main__":
    chain = build_rag_chain("./chroma_db", role ="hr")
    response = chain.invoke("What is the company finance condition?")
    if not response:
        print("I don't have access to that information or no matching documents were found.")
    print(response)





