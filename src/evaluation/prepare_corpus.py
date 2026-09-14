import argparse
import asyncio
import json
from pathlib import Path

from langchain_core.documents import Document

from src.ingestion.vector_store import VectorStoreManager
from src.utils.nlp_utils import segment_vietnamese


def load_contexts(dataset_path: Path) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()

    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            for context in row["ground_truth_context"]:
                normalized = context.strip()
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    contexts.append(normalized)

    return contexts


async def prepare_corpus(dataset_path: Path, collection_name: str) -> None:
    contexts = load_contexts(dataset_path)
    documents = [
        Document(
            page_content=segment_vietnamese(context),
            metadata={
                "original_text": context,
                "source": str(dataset_path),
                "page": None,
                "chunk_index": index,
                "chunk_length": len(context),
                "is_list": False,
            },
        )
        for index, context in enumerate(contexts)
    ]

    vector_store = VectorStoreManager()
    await vector_store.delete_collection(collection_name)
    await vector_store.create_collection(collection_name)
    await vector_store.upsert_documents(documents, collection_name)
    print(f"Indexed {len(documents)} unique contexts into '{collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build an evaluation corpus in Qdrant.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/rag_evaluation_dataset.jsonl"),
    )
    parser.add_argument("--collection", required=True)
    args = parser.parse_args()

    asyncio.run(prepare_corpus(args.dataset, args.collection))
