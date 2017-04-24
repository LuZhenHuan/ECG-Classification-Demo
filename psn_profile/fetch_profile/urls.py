from django.conf.urls import url
from . import views

app_name='fetch_profile'

urlpatterns = [
    url(r'^psn_profile/', views.home, name='home'),
    url(r'^query/ ', views.getProfile, name='getProfile')
]