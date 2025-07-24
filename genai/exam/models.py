from django.db import models
from pdfstore.models import PastQuestion
from pgvector.django import VectorField
class ExamSession(models.Model):
    EXAM_TYPE_CHOICES = (
        ('50', '50 Marks'),
        ('100', '100 Marks'),
    )
    exam_type = models.CharField(max_length=10, choices=EXAM_TYPE_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)
    start_time = models.DateTimeField(null=True, blank=True)  # NEW FIELD

    total_marks = models.FloatField()
    time_limit_minutes = models.IntegerField()
    completed = models.BooleanField(default=False)

    def __str__(self):
        return f"{self.get_exam_type_display()} Exam - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

# class ExamSession(models.Model):
#     EXAM_TYPE_CHOICES = (
#         ('50', '50 Marks'),
#         ('100', '100 Marks'),
#     )
#     exam_type = models.CharField(max_length=10, choices=EXAM_TYPE_CHOICES)
#     created_at = models.DateTimeField(auto_now_add=True)
#     total_marks = models.FloatField()
#     time_limit_minutes = models.IntegerField()
#     completed = models.BooleanField(default=False)

#     def __str__(self):
#         return f"{self.get_exam_type_display()} Exam - {self.created_at.strftime('%Y-%m-%d %H:%M')}"

class ExamQuestion(models.Model):
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name='questions')
    question_text = models.TextField()
    correct_answer = models.TextField(null=True, blank=True)  # Filled after AI grading (if needed)
    marks = models.FloatField(default=5.0)

    def __str__(self):
        return self.question_text[:80]

class StudentAnswer(models.Model):
    session = models.ForeignKey(ExamSession, on_delete=models.CASCADE, related_name='answers')
    question = models.ForeignKey(ExamQuestion, on_delete=models.CASCADE)
    answer_text = models.TextField(blank=True)
    skipped = models.BooleanField(default=False)
    marks_awarded = models.FloatField(default=0.0)

    def __str__(self):
        return f"Ans for: {self.question.question_text[:50]}"
