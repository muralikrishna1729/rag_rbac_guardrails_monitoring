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
from app.guardrails.guardrail import check_input_guardrail,check_output_guardrail

load_dotenv()

text_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_rag_chain(persist_directory: str, role:str):
    if not os.path.exists(persist_directory):
        raise FileNotFoundError(f"Vector database not found at {persist_directory}")
    vectorstore = Chroma(persist_directory= persist_directory ,embedding_function= text_embedding)

    search_kwargs = {"k":3}
    if role != 'admin':
        search_kwargs["filter"] = {"$or" : [{"role":role},{"role":"general"}]}
    else:
        print("Admin access granted: Searching all departments.")

    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)   

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
        You are a professional and friendly Company AI Assistant.

        Instructions:
            1. GREETINGS: If the user says 'Hi' or 'Hello', reply: "Hello! I'm your company assistant. How can I help you today?"
            2. IDENTITY: If asked 'Who are you?'or 'How do you doing' like that reply: "I am a RAG-powered assistant specialized in company policies.I'm Here to assist you"
            3. EXCEPTIONS: 
               - HR Email: hr@company.com
               - Leave Portal: https://portal.company.com/leave
            4. RAG: For all other questions, use the context below. 
            5. FALLBACK: If info is not in context or exceptions, say "I don't have that information."

        STRICT RULES:
            1. Use the context below to answer questions.
            2. If the answer is NOT in the context AND not in the 'EXCEPTIONS' above, 
               say: "I'm sorry, I don't have that information in our records."
            3. Do not make up facts.

        Context: {context}
        Question: {question}
        Answer:
        """
    )

    # chain = ({"context": retriever | format_docs ,"question":RunnablePassthrough()}| prompt | ChatHuggingFace(llm=llm) | StrOutputParser())
    chain = ({"context": retriever | (lambda docs:check_access(docs,role)) ,"question":RunnablePassthrough()}| prompt | llm | StrOutputParser())

    return chain, retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def check_access(docs,role):
    if not docs:
        return "I don't have access to that information or no matching documents were found."
    else:
        return format_docs(docs)

def ask_question(username:str, question:str):
    role = get_user_role(username)
    if not role:
        return "Access Denied: User not recognized."
    violation = check_input_guardrail(question)
    if violation:
        return violation
    
    chain,_ = build_rag_chain("./chroma_db",role = role)
    raw_response = chain.invoke(question)
    safe_response = check_output_guardrail(raw_response)
    return safe_response


if __name__== "__main__":
    response = ask_question("alice", "How many candidates are onboarded in this year to the company?")
    if not response:
        print("I don't have access to that information or no matching documents were found.")
    print(response)






