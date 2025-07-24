from django.contrib import admin
from .models import PastQuestion,SyllabusTopic,NoteContent
admin.site.register(SyllabusTopic)
admin.site.register(NoteContent)
admin.site.register(PastQuestion)