from langchain_community.llms import Ollama
from textblob import TextBlob
from django.db import transaction
from pdfstore.utils import get_similar_questions
from exam.models import ExamSession, ExamQuestion

def generate_questions_from_chunks(chunks, llm_model="mistral:7b", num_questions=5):
    prompt = (
        "Generate distinct exam questions based on the following excerpts:\n\n"
        + "\n\n".join(chunks)
        + "\n\nPlease list the questions clearly."
    )
    ollama_llm = Ollama(model=llm_model)
    output = ollama_llm.invoke(prompt)
    questions = [q.strip() for q in output.split('\n') if q.strip()]
    return questions[:num_questions]

@transaction.atomic
def generate_exam_with_llm_from_query(query_text, exam_type='50'):
    marks_required = 50 if exam_type == '50' else 100
    per_question_marks = 5
    num_questions = marks_required // per_question_marks

    similar_objs = get_similar_questions(query_text, top_k=num_questions)
    chunks = [obj.question_text for obj in similar_objs]

    generated_questions = generate_questions_from_chunks(chunks, num_questions=num_questions)

    session = ExamSession.objects.create(
        exam_type=exam_type,
        total_marks=marks_required,
        time_limit_minutes=int((marks_required / 100) * 180),
    )
    for q_text in generated_questions:
        ExamQuestion.objects.create(
            session=session,
            question_text=q_text,
            marks=per_question_marks,
        )
    return session