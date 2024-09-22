from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('choose/', views.choose, name='choose'),
    path('make/', views.make, name='make'),
    path('make_ch1/', views.make_ch1, name='make_ch1'),
    path('choose_ch1/', views.choose_ch1, name='choose_ch1'),
    path('chat/', views.chat, name='chat'),
    path('select_number_of_people/', views.select_number_of_people, name='select_number_of_people'),
    path('query_view/', views.query_view, name='query_view'),
    path('voice_separation/', views.voice_separation, name='voice_separation'),
    path('confirm_voice/', views.confirm_voice, name='confirm_voice'),
    path('select_speaker/', views.select_speaker, name='select_speaker'),
    path('api/summarize/', views.summarize_text, name='summarize_text'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
