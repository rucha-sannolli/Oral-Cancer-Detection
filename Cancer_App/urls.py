from django.urls import path,include
from . import views

from django.contrib.auth import views as auth_views

urlpatterns = [
    path("home/",views.index),
    path("reg/",views.reg),
    path("login/",views.login_view,name="login"),
    path("predict/",views.predict,name="predict"),
    path("result/",views.result,name="result"),
    # path('result_page/', views.result_page, name='result_page'),

    path('predict_stage/', views.predict_stage, name='predict_stage'), 

    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),

    path('contact/', views.contact_view, name='contact'),
]
