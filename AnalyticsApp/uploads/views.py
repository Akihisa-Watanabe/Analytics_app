import logging
import stripe
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
from .analysis.remove_noise import remove_noise
from .analysis.find_peaks import find_peaks

logger = logging.getLogger(__name__)


class IndexView(LoginRequiredMixin, View):
    def get(self, request, *args, **kwargs):
        queryset = Upload.objects.filter(file__contains='files/upload/{0}/'.format(self.request.user.id)).values()
        #first=queryset.values_list("id",flat=True)
        #first_id = first[0]*(-1)
        print(queryset[0])

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
        data= pd.read_csv(path, names=['ax','ay','az','gx','gy','gz'])
        data = pd.Series(data['gx'], dtype='int')
        originalData = np.array(data)
        N=len(originalData)
        dt = 0.005

        filteredData = remove_noise(originalData,dt,N)
        maximal_idx=find_peaks(filteredData,N)
        time_per_step =[]
        for i in range(1,len(maximal_idx)):
            time_per_step.append(maximal_idx[i]-maximal_idx[i-1])


        x=[i for i in range(len(time_per_step))]



        t = np.arange(0, N*dt,dt) #時間軸
        plot1 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="angular velocity",plot_width=800,plot_height=600)
        plot1.line(t,filteredData)

        plot2 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="angular velocity",plot_width=800,plot_height=600)
        plot2.line(x,time_per_step)
        script, div = components([plot1,plot2])
        return render_to_response( 'upload/file_detail.html',{'script' : script , 'div' : div} )



detail = DetailView.as_view()
