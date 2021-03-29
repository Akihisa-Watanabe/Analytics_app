import logging
import stripe
from django.http import HttpResponse
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q
from django.shortcuts import render, get_object_or_404, redirect, reverse, render_to_response
from django.views.generic import View, UpdateView
from .forms import UploadForm
from .models import Upload
from django.contrib.auth import get_user_model
from bokeh.plotting import figure, output_file, show
from  bokeh.embed  import components
import numpy as np
import pandas as pd
from .analysis.SSA import graph_plot 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64

logger = logging.getLogger(__name__)


class IndexView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        queryset = Upload.objects.filter(file__contains='files/upload/{0}/'.format(self.request.user.id)).values()
        #first=queryset.values_list("id",flat=True)
        #first_id = first[0]*(-1)
        #print(queryset[0])

        for i in range(len(queryset)):
            queryset[i].update(file_id=i)

        context = {
            'queryset': queryset,
        }

        return render(request, 'upload/file_list.html', context )


index = IndexView.as_view()

class UploadView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        context = {'form': UploadForm()}
        return render(request, 'upload/upload_form.html', context)

    def post(self, request, *args , **kwargs):
        form = UploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return render(request, 'upload/upload_form.html', {'form': form})

        upload = form.save(commit=False)
        upload.submitter = request.user
        upload.save()#ここでファイルがアップロードされる
        User = get_user_model()
        User = User.objects.get(username=self.request.user)
        User.upload_count +=1
        User.save()

        return redirect(reverse('uploads:index'))

upload = UploadView.as_view()

#ここでグラフとか表示して分析
class DetailView(LoginRequiredMixin, View):
    def get(self, request, file_id, *args, **kwargs):
        path=settings.MEDIA_ROOT+'/files/upload/{0}/{1}'.format(self.request.user.id, file_id)
        graph = graph_plot(path)
        graph.create_fig(option=1)
        svg1 = graph.plt2svg()
        plt.cla()  
        graph.create_fig(option=2)
        svg2 = graph.plt2svg()
        plt.cla()  
        baton_time = graph.create_fig(option=3)
        svg3 = graph.plt2svg()
        plt.cla()  
        context = {"graph1":svg1, "graph2":svg2, "graph3":svg3, "baton_time":baton_time}
        #response = HttpResponse(svg3, content_type='image/svg+xml')
        #return response
        return render(request, "upload/file_detail.html", context)



detail = DetailView.as_view()
