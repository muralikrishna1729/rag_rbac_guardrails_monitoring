import os
import pandas as pd
from dotenv import load_dotenv

from langchain_community.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextRecall, ContextPrecision

from datasets import Dataset
from app.pipeline.rag_chain import build_rag_chain

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH    = "./resources/data"
TESTSET_PATH = "./resources/test_data/synthetic_rag_testset.csv"
CHROMA_PATH  = "./chroma_db"


# ── Shared wrappers (created once, reused everywhere) ─────────────────────────
def get_ragas_llm():
    return LangchainLLMWrapper(
        ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
    )

def get_ragas_embeddings():
    return LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )


def generate_test_data(data_path: str) -> pd.DataFrame:
    """Load documents and generate a synthetic Q&A testset using Ragas."""

    md_loader = DirectoryLoader(
        data_path,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )
    csv_loader = DirectoryLoader(
        data_path,
        glob="**/*.csv",
        loader_cls=CSVLoader
    )

    documents = md_loader.load() + csv_loader.load()
    print(f"Loaded {len(documents)} documents for test generation.")

    # Use Gemini for generation (better quality than Groq for this task)
    generation_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.7
        )
    )

    embeddings = get_ragas_embeddings()

    generator = TestsetGenerator(
        llm=generation_llm,
        embedding_model=embeddings
    )

    run_config = RunConfig(
        max_retries=10,
        max_wait=90,
        timeout=120,
    )

    testset = generator.generate_with_langchain_docs(
        documents,
        testset_size=10,
        run_config=run_config
    )

    test_df = testset.to_pandas()
    print(f"\nGenerated {len(test_df)} test samples.")
    print(test_df[["question", "ground_truth"]].head())

    os.makedirs(os.path.dirname(TESTSET_PATH), exist_ok=True)
    test_df.to_csv(TESTSET_PATH, index=False)
    print(f"Testset saved to {TESTSET_PATH}")

    return test_df


def collect_rag_results(test_df: pd.DataFrame) -> dict:
    """Run the RAG chain on each question and collect answers + contexts."""

    rag_chain, retriever = build_rag_chain(
        persist_directory=CHROMA_PATH,
        role="admin"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    questions    = test_df["question"].tolist()
    ground_truths = test_df["ground_truth"].tolist() if "ground_truth" in test_df.columns else None

    answers  = []
    contexts = []

    print(f"\nRunning RAG chain on {len(questions)} questions...")
    for i, question in enumerate(questions):
        retrieved_docs = retriever.invoke(question)
        answer         = rag_chain.invoke(question)

        answers.append(answer)
        contexts.append([doc.page_content for doc in retrieved_docs])

        print(f"  [{i+1}/{len(questions)}] Done: {question[:60]}...")

    data_dict = {
        "question": questions,
        "answer":   answers,
        "contexts": contexts,
    }
    if ground_truths:
        data_dict["ground_truth"] = ground_truths

    return data_dict


def evaluate_rag_chain(test_df: pd.DataFrame):
    """Evaluate collected results using Ragas faithfulness, relevancy, recall, precision."""

    data_dict   = collect_rag_results(test_df)
    eval_dataset = Dataset.from_dict(data_dict)

    ragas_llm        = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings),
        ContextRecall(llm=ragas_llm),
        ContextPrecision(llm=ragas_llm),
    ]

    print("\nEvaluating RAG performance with Ragas...")
    try:
        scores = evaluate(
            dataset=eval_dataset,
            metrics=metrics
        )
    except Exception as e:
        print(f"Evaluation failed: {e}")
        return

    scores_df = scores.to_pandas()

    print("\n─── RAGAS EVALUATION RESULTS (per question) ───────────────────")
    display_cols = ["question", "faithfulness", "answer_relevancy", "context_recall", "context_precision"]
    available    = [c for c in display_cols if c in scores_df.columns]
    print(scores_df[available].to_string(index=False))

    print("\n─── AVERAGE SCORES ─────────────────────────────────────────────")
    for col in ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]:
        if col in scores_df.columns:
            print(f"  {col:<22}: {scores_df[col].mean():.2f}")

    results_path = "./resources/test_data/ragas_results.csv"
    scores_df.to_csv(results_path, index=False)
    print(f"\nFull results saved to {results_path}")



if __name__ == "__main__":
    print("Starting RAG evaluation...\n")

    if os.path.exists(TESTSET_PATH):
        print(f"Existing testset found at {TESTSET_PATH}. Loading...")
        test_df = pd.read_csv(TESTSET_PATH)
        print(f"Loaded {len(test_df)} test samples.")
    else:
        print("No testset found. Generating new one (this may take a few minutes)...")
        test_df = generate_test_data(DATA_PATH)

    evaluate_rag_chain(test_df)