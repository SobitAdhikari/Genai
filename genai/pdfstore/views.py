
import io
import logging
from django.shortcuts import render,redirect
from django.db import transaction
from .forms import PDFUploadForm
from .models import PastQuestion
from .utils import (
    extract_text_from_pdf,
    extract_questions_with_ollama,   # LLM-based
    chunk_questions,
    extract_topics_from_text,
    get_embedding,
)

logger = logging.getLogger(__name__)


def upload_past_questions(request):
    """
    Handles upload of past question PDFs.
    Uses Mistral 7B via Ollama for extraction,
    embeds with mpnet, stores in pgvector DB.
    """
    message = ""

    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            past_questions_pdf_file = request.FILES.get('past_questions_pdf')

            if not past_questions_pdf_file:
                message = " Please upload a past questions PDF."
                return render(request, 'pdfstore/upload.html', {'form': form, 'message': message})

            try:
                with transaction.atomic():
                    past_q_text = extract_text_from_pdf(io.BytesIO(past_questions_pdf_file.read()))
                    questions_list = extract_questions_with_ollama(past_q_text)

                    if not questions_list:
                        message = " No questions extracted from the PDF."
                    else:
                        question_chunks = chunk_questions(questions_list, chunk_size=300, chunk_overlap=50)

                        for chunk in question_chunks:
                            PastQuestion.objects.create(
                                question_text=chunk,
                                embedding=get_embedding(chunk),
                            )

                        message = f" {len(question_chunks)} question chunks processed and stored successfully."

            except Exception as e:
                logger.error(f"[ERROR] Past questions processing failed: {e}")
                message = (
                    f" Error: {e}. Ensure Ollama is running and SentenceTransformers are properly configured."
                )
        else:
            message = "Please correct the errors in the form."
    else:
        form = PDFUploadForm()

    return render(request, 'pdfstore/upload.html', {'form': form, 'message': message})




from django.shortcuts import render
from django.db import transaction
from .forms import NoteUploadForm
from .utils import process_and_save_notes

def upload_notes(request):
    """
    View to upload notes PDF, chunk it, embed it, and store in vector DB.
    """
    message = ""

    if request.method == 'POST':
        form = NoteUploadForm(request.POST, request.FILES)
        if form.is_valid():
            note_pdf = request.FILES['note_pdf']

            try:
                with transaction.atomic():
                    saved_count = process_and_save_notes(note_pdf)
                    message = f" {saved_count} note chunks saved successfully."

            except Exception as e:
                print(f"[ERROR] Notes upload failed: {e}")
                message = f" Error: {e}"

    else:
        form = NoteUploadForm()

    return render(request, 'pdfstore/upload_notes.html', {'form': form, 'message': message})

# For topics
from .forms import SyllabusUploadForm
from .models import SyllabusTopic

from pdfstore.utils import get_embedding

def upload_syllabus_topics(request):
    if request.method == "POST":
        form = SyllabusUploadForm(request.POST)
        if form.is_valid():
            syllabus_text = form.cleaned_data['syllabus_text']
            default_hours = form.cleaned_data['default_hours']
            topics = extract_topics_from_text(syllabus_text)

            saved = 0
            skipped = 0

            for topic in topics:
                if SyllabusTopic.objects.filter(topic_name__iexact=topic).exists():
                    skipped += 1
                    continue

                new_topic = SyllabusTopic.objects.create(
                    topic_name=topic,
                    subtopic_content=topic,
                    hours=default_hours
                )

                try:
                    embedding = get_embedding(topic)
                    new_topic.embedding = embedding
                    new_topic.save()
                except Exception as e:
                    print(f" Embedding failed for {topic}: {e}")

                saved += 1

            message = f" {saved} topics saved and vectorized.  {skipped} skipped (duplicates)."
            return render(request, "pdfstore/upload_syllabus.html", {"form": form, "message": message})
    else:
        form = SyllabusUploadForm()

    return render(request, "pdfstore/upload_syllabus.html", {"form": form})
