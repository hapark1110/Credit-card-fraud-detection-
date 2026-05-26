from django.urls import path
from . import views

urlpatterns = [
    path('',                views.index,            name='index'),
    path('single/',         views.single_predict,   name='single_predict'),
    path('upload/',         views.upload_csv,        name='upload_csv'),
    path('dashboard/',      views.dashboard,         name='dashboard'),
    path('live/',           views.live_monitor,      name='live_monitor'),

    # AJAX / SSE endpoints
    path('api/predict/',        views.api_predict,       name='api_predict'),
    path('api/explain/',        views.api_explain,       name='api_explain'),
    path('api/upload-analyze/', views.api_upload_analyze,name='api_upload_analyze'),
    path('api/live-stream/',    views.api_live_stream,   name='api_live_stream'),
]
