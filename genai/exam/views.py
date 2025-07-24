from django.shortcuts import render, redirect, get_object_or_404
from django.utils.timezone import now
from exam.models import ExamSession, StudentAnswer
from exam.utils import generate_exam_with_llm_from_query
from pdfstore.models import NoteContent
from langchain_community.llms import Ollama
from .forms import ExamSetupForm, AnswerForm
import re
from pdfstore.models import NoteContent
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
import torch


def select_exam(request):
    if request.method == 'POST':
        form = ExamSetupForm(request.POST)
        if form.is_valid():
            exam_type = int(form.cleaned_data['exam_type'])  # Convert to int
            query_text = "artificial intelligence, machine learning, neural networks"

            session = generate_exam_with_llm_from_query(query_text=query_text, exam_type=exam_type)

            # Dynamic time calculation: (marks / 100) * 180
            session.time_limit_minutes = round((exam_type / 100) * 180)
            session.total_marks = exam_type
            session.save()

            return redirect('exam:take_exam', session_id=session.id, index=0)
    else:
        form = ExamSetupForm()

    return render(request, 'exam/select_exam.html', {'form': form})




embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def take_exam(request, session_id, index):
    session = get_object_or_404(ExamSession, id=session_id)
    questions = list(session.questions.all())
    index = int(index)

    if session.start_time is None:
        session.start_time = now()
        session.save()

    elapsed_time = (now() - session.start_time).total_seconds() / 60
    time_remaining = max(session.time_limit_minutes - elapsed_time, 0)

    if time_remaining <= 0 or index >= len(questions):
        return redirect('exam:exam_summary', session_id=session.id)

    question = questions[index]

    if request.method == 'POST':
        form = AnswerForm(request.POST)
        if form.is_valid():
            answer_text = form.cleaned_data['answer_text']
            skip = 'skip' in request.POST

            StudentAnswer.objects.update_or_create(
                session=session,
                question=question,
                defaults={
                    'answer_text': answer_text if not skip else '',
                    'skipped': skip,
                }
            )

            if 'submit_exam' in request.POST:
                return redirect('exam:submit_exam', session_id=session.id)

            if 'next' in request.POST:
                next_index = index + 1
            elif 'prev' in request.POST:
                next_index = max(index - 1, 0)
            elif 'skip' in request.POST:
                next_index = index + 1
            else:
                next_index = index + 1

            return redirect('exam:take_exam', session_id=session.id, index=next_index)
    else:
        form = AnswerForm()

    is_last_question = (index == len(questions) - 1)

    return render(request, 'exam/take_exam.html', {
        'session': session,
        'question': question,
        'index': index,
        'total': len(questions),
        'time_remaining': round(time_remaining, 2),
        'form': form,
        'is_last_question': is_last_question,
    })



embedding_model = SentenceTransformer("all-mpnet-base-v2")

from django.views.decorators.csrf import csrf_exempt

from django.http import HttpResponseBadRequest

@csrf_exempt
def submit_exam(request, session_id):
    session = get_object_or_404(ExamSession, id=session_id)

    if session.completed:
        return redirect('exam_summary', session_id=session.id)

    if request.method == 'GET':
      
        answers = StudentAnswer.objects.filter(session=session).select_related('question')
        
        llm = Ollama(model="mistral:7b")
        total_marks = 0
        similarity_threshold = 0.4

        note_embeddings = [
            (note, torch.tensor(note.embedding).unsqueeze(0))
            for note in NoteContent.objects.all()
        ]

        for ans in answers:
            if ans.skipped:
                ans.marks_awarded = 0.0
                ans.save()
                continue

            ans_emb_tensor = torch.tensor(embedding_model.encode(ans.answer_text)).unsqueeze(0)
            best_score = 0.0

            for note, note_tensor in note_embeddings:
                score = cos_sim(ans_emb_tensor, note_tensor).item()
                if score > best_score:
                    best_score = score

            if best_score > similarity_threshold:
                ans.marks_awarded = ans.question.marks
            else:
                prompt = (
                    f"You are a strict exam evaluator.\n\n"
                    f"Evaluate the student's answer based on the following question. "
                    f"Assign a score **out of {ans.question.marks}**. "
                    f"Only respond with a **number**.\n\n"
                    f"Question:\n{ans.question.question_text}\n\n"
                    f"Student's Answer:\n{ans.answer_text}"
                )
                try:
                    result = llm.invoke(prompt).strip()
                    mark = float(result)
                    ans.marks_awarded = min(max(mark, 0.0), ans.question.marks)
                except:
                    ans.marks_awarded = 0.0

            ans.save()
            total_marks += ans.marks_awarded

        session.completed = True
        session.total_marks = round(total_marks, 2)
        session.save()

        return redirect('exam_summary', session_id=session.id)

    return HttpResponseBadRequest("Invalid request method.")

def exam_summary(request, session_id):
    session = get_object_or_404(ExamSession, id=session_id)
    answers = StudentAnswer.objects.filter(session=session).select_related('question')

    return render(request, 'exam/exam_summary.html', {
        'session': session,
        'answers': answers,
        'total_marks': session.total_marks
    })

