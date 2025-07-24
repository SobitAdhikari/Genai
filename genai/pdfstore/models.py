# pdfstore/models.py

from django.db import models
from pgvector.django import VectorField

class PastQuestion(models.Model):
    """
    Model for storing past exam questions with vector embeddings.
    """
    question_text = models.TextField(help_text="The cleaned text of the past question (chunked if needed).")
    embedding = VectorField(dimensions=768, help_text="Vector embedding of the question text .")

    def __str__(self):
        return f"Q: {self.question_text[:80]}..."


# addded for note content

# working
class SyllabusTopic(models.Model): 
    topic_name = models.CharField(max_length=255, unique=True)
    subtopic_content = models.TextField()
    hours = models.FloatField()
    embedding = VectorField(dimensions=768, null=True, blank=True)  # Allows topics to be embedded later if needed

    def __str__(self):
        return self.topic_name

class NoteContent(models.Model):
    subtopic = models.ForeignKey(
        SyllabusTopic,
        on_delete=models.CASCADE,
        related_name='note_contents',
        null=True,    # Allow NULL in DB
        blank=True    # Allow blank in forms/admin
    )
    note_text = models.TextField()
    embedding = VectorField(dimensions=768)

    def __str__(self):
        return (f"{self.subtopic.topic_name if self.subtopic else 'Unlinked'} - "
                f"{self.note_text[:50]}...")

    


    # for topics
