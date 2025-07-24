# exam/urls.py

from django.urls import path
from . import views
from .views import submit_exam, exam_summary

app_name = 'exam'

urlpatterns = [
    path('', views.select_exam, name='select_exam'),
    path('take/<int:session_id>/<int:index>/', views.take_exam, name='take_exam'),
    # path('summary/<int:session_id>/', views.exam_summary, name='exam_summary'),
    path('exam/<int:session_id>/submit/', submit_exam, name='submit_exam'),
    path('exam/<int:session_id>/summary/', exam_summary, name='exam_summary'),

]
