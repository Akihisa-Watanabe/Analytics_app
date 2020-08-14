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
        axis =['ax','ay','az','gx','gy','gz','mx','my','mz']
        data= pd.read_csv(path, names=axis)
        ax=pd.Series(data['ax'], dtype='int')
        ay=pd.Series(data['ay'], dtype='int')
        az=pd.Series(data['az'], dtype='int')
        gx=pd.Series(data['gx'], dtype='int')
        gy=pd.Series(data['gy'], dtype='int')
        gz=pd.Series(data['gz'], dtype='int')
        mx=pd.Series(data['mx'], dtype='int')
        my=pd.Series(data['my'], dtype='int')
        mz=pd.Series(data['mz'], dtype='int')

        ax=np.array(ax)
        ay=np.array(ay)
        az=np.array(az)
        gx=np.array(gx)
        gy=np.array(gy)
        gz=np.array(gz)
        mx=np.array(mx)
        my=np.array(my)
        mz=np.array(mz)

        


        dt = 0.005
        N = len(ax)

        t = np.arange(0, N*dt,dt) #時間軸
        plot1 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="ax",plot_width=1300,plot_height=700)
        plot1.line(t,ax)
        

        plot2 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="ay",plot_width=1300,plot_height=700)
        plot2.line(t,ay)

        plot3 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="az",plot_width=1300,plot_height=700)
        plot3.line(t,az)

        plot4 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="gx",plot_width=1300,plot_height=700)
        plot4.line(t,gx)

        plot5 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="gy",plot_width=1300,plot_height=700)
        plot5.line(t,gy)

        plot6 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="gz",plot_width=1300,plot_height=700)
        plot6.line(t,gz)

        plot7 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="mx",plot_width=1300,plot_height=700)
        plot7.line(t,mx)

        plot8 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="my",plot_width=1300,plot_height=700)
        plot8.line(t,my)

        plot9 = figure(x_axis_label="time",x_axis_type="datetime",y_axis_label="mz",plot_width=1300,plot_height=700)
        plot9.line(t,mz)

        script, div = components([plot1,plot2, plot3,plot4,plot5,plot6,plot7,plot8,plot9])
        return render_to_response( 'upload/file_detail.html',{'script' : script , 'div' : div} )



detail = DetailView.as_view()
