# pdfstore/forms.py

from django import forms

class PDFUploadForm(forms.Form):
    """
    Form for uploading only past questions PDF.
    """
    past_questions_pdf = forms.FileField(
        label='Upload Past Questions PDF',
        required=True,
        help_text="Upload the PDF containing past exam questions."
    )

    def clean(self):
        """
        Custom clean method to validate the form.
        """
        cleaned_data = super().clean()
        past_questions_pdf = cleaned_data.get('past_questions_pdf')

        if not past_questions_pdf:
            raise forms.ValidationError("Please upload a past questions PDF file.")
        
        return cleaned_data


# forms for note upload 
# pdfstore/forms.py

from django import forms

class NoteUploadForm(forms.Form):
    """
    Upload Notes PDF separately for vector storage.
    """
    note_pdf = forms.FileField(label='Upload Notes PDF', required=True,
                               help_text="Upload the PDF containing detailed notes for subtopics.")




# topic uploading

from django import forms

class SyllabusUploadForm(forms.Form):
    """
    Upload a text blob containing syllabus topics (one per line, numbered or plain).
    """
    syllabus_text = forms.CharField(
        widget=forms.Textarea(attrs={"rows": 10, "placeholder": "Paste numbered topics here..."}),
        help_text="Paste topics like:\n1. Topic One\n2. Topic Two\n..."
    )

    default_hours = forms.FloatField(
        label="Default Hours per Topic",
        initial=3.0,
        help_text="Default allocated hours for each topic (can be edited later)."
    )
