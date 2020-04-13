from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.auth import get_user_model
import os


def get_file_path(instance,filename):
    User = get_user_model()
    User = User.objects.get(username=str(instance.submitter))
    return 'files/upload/{0}/{1}'.format(instance.submitter.id, User.upload_count)


def validate_is_csv(value):
    ext = os.path.splitext(value.name)[1]
    if not ext.lower() in ['.csv']:
        raise ValidationError('Only csv files are availables.')


class Upload(models.Model):
    """走者(投稿者)モデル"""

    class Meta(object):
        db_table = 'upload'

    #name = models.ForeignKey(get_user_model(), on_delete=models.PROTECT, verbose_name='投稿者')
    file = models.FileField(verbose_name='ファイル', upload_to=get_file_path, validators=[validate_is_csv])
    upload_date = models.DateField(verbose_name='アップロード日', auto_now=True)
