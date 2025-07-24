# exam/forms.py

from django import forms

class ExamSetupForm(forms.Form):
    EXAM_CHOICES = (
        ('50', '50 Marks Exam (1.5 hrs)'),
        ('100', '100 Marks Exam (3 hrs)'),
    )

    exam_type = forms.ChoiceField(
        choices=EXAM_CHOICES,
        widget=forms.RadioSelect,
        label="Select Exam Type",
        help_text="Choose between 50 or 100 marks exam."
    )

class AnswerForm(forms.Form):
    answer_text = forms.CharField(
        widget=forms.Textarea(attrs={
            'rows': 6,
            'placeholder': 'Type your answer here...',
            'style': 'width:100%; padding:10px; font-size:1em;'
        }),
        required=False,
        label='Your Answer'
    )
