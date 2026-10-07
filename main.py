from src.classification.classifier import classify_complaint
from src.generation.generator import generate_resolution_steps
from src.retrieval.retriever import retrieve_relevant_records
from src.tui.app import run_app

def main():
    run_app()

if __name__ == "__main__":
    main()