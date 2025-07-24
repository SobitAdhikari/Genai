# pdfstore/urls.py

from django.urls import path
from . import views

app_name = 'pdfstore' # Namespace for your app's URLs

urlpatterns = [
    path('upload/', views.upload_past_questions, name='upload_pdf'),
    path('uploadnotes/', views.upload_notes, name='upload_notes'),
    path('upload-syllabus/', views.upload_syllabus_topics, name='upload_syllabus'),
]