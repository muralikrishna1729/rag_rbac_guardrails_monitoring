import re 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()


PII_PATTERNS = [
    r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',  # email
    r'\b\d{10}\b',                                      # phone
    r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'                  # aadhaar
]
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

def detect_pii(text:str)->bool:
    for pattern in PII_PATTERNS:
        if re.search(pattern, text):
            return True 
    return False


def check_scope(question: str) -> bool:
    prompt = ChatPromptTemplate.from_template(
        """
        Is this question related to company business such as HR, finance, marketing, or engineering?
        Question: {question}
        Answer only YES or NO.
        """
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question":question})
    return "YES" in result.upper()


def check_input_guardrail(question: str)->str:
    if detect_pii(question):
        return "Your query contains sensitive personal information. Please remove it."
    if not check_scope(question):
        return "I can only answer questions related to company operations."
    return None


def check_output_guardrail(response: str) -> str:
    for pattern in PII_PATTERNS:
        response = re.sub(pattern, "[REDACTED]", response)
    return response

if __name__ == "__main__":
    print(check_input_guardrail("What is john@company.com salary?"))
    print(check_input_guardrail("Who won IPL 2024?"))
    print(check_input_guardrail("What is the leave policy?"))
    print(check_output_guardrail("Contact HR at hr@company.com or call 9876543210"))





