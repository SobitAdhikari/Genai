import re
from typing import List, Tuple
from django.db import transaction
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from .models import PastQuestion
llm = Ollama(model="mistral")  
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2")  # 768 dimension

def get_embedding(text: str) -> list[float]:
    """
    Generate 768-dimensional embedding using SentenceTransformers.
    """
    try:
        embedding = model.encode(text, normalize_embeddings=True) 
        return embedding.tolist() 
    except Exception as e:
        raise RuntimeError(f"[Embedding Error] {e}")


import logging
logger = logging.getLogger(__name__)



# PDF Extraction

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract raw text from a PDF file.
    """
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)


# LLM-based Question Extraction

def extract_questions_with_ollama(text: str, model: str = "mistral:7b") -> List[str]:
    """
    Use Ollama Mistral to extract exam questions from raw text.
    """
    prompt = (
        "Extract all exam questions from the following text. "
        "For each question, start the line with 'Question:'.\n\n"
        f"{text}"
    )

    try:
        ollama_llm = Ollama(model=model)
        raw_output = ollama_llm.invoke(prompt)
    except Exception as e:
        logger.error(f"[Ollama Extraction Error] {e}")
        raise RuntimeError(f"Ollama inference failed: {e}")

    # Split questions based on "Question:" delimiter
    questions = [q.strip() for q in raw_output.split("Question:") if q.strip()]

    logger.info(f"LLM extracted {len(questions)} questions from PDF.")
    return questions


# Chunking

def chunk_questions(questions: List[str], chunk_size: int = 300, chunk_overlap: int = 50) -> List[str]:
    """
    Chunk long questions for better embedding handling.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks = []
    for q in questions:
        chunks = splitter.split_text(q)
        all_chunks.extend(chunks)

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


# Save to DB 

def save_questions_to_db(chunks: List[str]) -> None:
    """
    Save question chunks and embeddings to Postgres (pgvector).
    """
    with transaction.atomic():
        for chunk in chunks:
            embedding = get_embedding(chunk)
            PastQuestion.objects.create(
                question_text=chunk,
                embedding=embedding  # pgvector accepts list of floats directly
            )
    logger.info(f" Saved {len(chunks)} question chunks with embeddings to Postgres (pgvector).")

from pdfstore.models import PastQuestion
from pgvector.django import L2Distance
from sentence_transformers import SentenceTransformer

# added for retrieval 
embedding_model = SentenceTransformer("all-mpnet-base-v2")  # Must match save time

def get_similar_questions(query_text: str, top_k: int = 10):
    query_embedding = embedding_model.encode(query_text).tolist()
    similar_questions = (
        PastQuestion.objects
        .annotate(distance=L2Distance('embedding', query_embedding))
        .order_by('distance')[:top_k]
    )
    return list(similar_questions)

# upto here 

#  Main Pipeline

def process_past_questions_pdf(pdf_file) -> None:
    """
    End-to-end pipeline:
    Extract text -> Use LLM to get questions -> Chunk -> Embed -> Store
    """
    logger.info("Starting LLM-based past question processing pipeline...")

    pdf_text = extract_text_from_pdf(pdf_file)
    questions = extract_questions_with_ollama(pdf_text)

    if not questions:
        logger.warning("No questions extracted by LLM.")
        return

    chunks = chunk_questions(questions)
    save_questions_to_db(chunks)

    logger.info("Past questions pipeline completed successfully.")


    # to process notes

from .models import NoteContent, SyllabusTopic


# working  extract topics
# def extract_topics_with_ollama(text: str, model: str = "mistral:7b") -> List[str]:
#     """
#     Use Ollama to extract syllabus topics from raw text.
#     """
#     prompt = (
#         "Extract all syllabus topics from the following text. "
#         "Each topic should start with 'Topic:'.\n\n"
#         "1 Introduction to Artificial Intelligence\n"
#         "1.1 Intelligence\n"
#         "1.1.1 Types of Intelligence\n\n"
#         "Extract only numbered topics, no extra text.\n\n"
#         f"{text}"
#         f"{text}"
#     )

#     try:
#         ollama_llm = Ollama(model=model)
#         raw_output = ollama_llm.invoke(prompt)
#     except Exception as e:
#         logger.error(f"[Ollama Topic Extraction Error] {e}")
#         raise RuntimeError(f"Ollama inference failed: {e}")

#     topics = [t.strip() for t in raw_output.split("Topic:") if t.strip()]
#     logger.info(f"LLM extracted {len(topics)} topics from PDF.")
#     return topics

def extract_topics_from_text(text: str) -> list:
    """
    Extracts topics from a numbered list text input.
    Removes numbering and trims whitespace.
    """
    lines = text.strip().split("\n")
    topics = []
    for line in lines:
        cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if cleaned:
            topics.append(cleaned)
    return topics

def extract_notes_with_ollama(text: str, model: str = "mistral:7b") -> List[str]:
    """
    Use Ollama to extract important notes from raw text.
    Each note should start with 'Note:'.
    """
    prompt = (
        "Extract all important notes or explanations from the following text. "
        "Each note should start with 'Note:'.\n\n"
        f"{text}"
    )

    try:
        ollama_llm = Ollama(model=model)
        raw_output = ollama_llm.invoke(prompt)
    except Exception as e:
        logger.error(f"[Ollama Note Extraction Error] {e}")
        raise RuntimeError(f"Ollama inference failed: {e}")

    notes = [n.strip() for n in raw_output.split("Note:") if n.strip()]
    logger.info(f"LLM extracted {len(notes)} notes from PDF.")
    return notes

def chunk_notes(notes: List[str], chunk_size=500, overlap=100) -> List[str]:
    """
    Chunk long notes into smaller pieces.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )

    all_chunks = []
    for note in notes:
        chunks = splitter.split_text(note)
        all_chunks.extend(chunks)

    logger.info(f"Total note chunks created: {len(all_chunks)}")
    return all_chunks

def save_notes_to_db(chunks: List[str], existing_topics: dict) -> None:
    """
    Save notes with embeddings and link to SyllabusTopic if matched.
    """
    with transaction.atomic():
        for chunk in chunks:
            embedding = get_embedding(chunk)

            linked_topic = None
            for topic_name, topic_obj in existing_topics.items():
                if topic_name in chunk.lower():
                    linked_topic = topic_obj
                    break

            NoteContent.objects.create(
                subtopic=linked_topic,  # Can be None if no match
                note_text=chunk,
                embedding=embedding
            )

    print(f" Saved {len(chunks)} note chunks with embeddings to Postgres (pgvector).")


def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    return "".join(page.extract_text() or "" for page in reader.pages)

def chunk_notes(text, chunk_size=500, overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_text(text)


def process_and_save_notes(note_pdf_file):
    text = extract_text_from_pdf(note_pdf_file)
    chunks = chunk_notes(text)

    # Try to get topics from DB
    existing_topics = {t.topic_name.lower(): t for t in SyllabusTopic.objects.all()}
    has_topics = bool(existing_topics)

    saved_count = 0

    # If no topics in DB, extract from text for fallback
    extracted_topics = extract_topics_from_text(text) if not has_topics else []

    for chunk in chunks:
        linked_topic = None

        if has_topics:
            # Try matching chunk with existing topics
            for topic_name, topic_obj in existing_topics.items():
                if topic_name in chunk.lower():
                    linked_topic = topic_obj
                    break

        embedding = get_embedding(chunk)

        if linked_topic:
            # Save with linked syllabus topic
            NoteContent.objects.create(
                subtopic=linked_topic,
                note_text=chunk,
                embedding=embedding
            )
            saved_count += 1
        else:
            # Save as unlinked chunk but still store embedding for retrieval
            NoteContent.objects.create(
                subtopic=None,  # Allow null in model or handle appropriately
                note_text=chunk,
                embedding=embedding
            )
            saved_count += 1
            print(f"[Info] Saved unlinked chunk: {chunk[:50]}...")

    return {
        "saved_chunks": saved_count,
        "topics_found": has_topics,
        "extracted_topics": extracted_topics if not has_topics else None
    }