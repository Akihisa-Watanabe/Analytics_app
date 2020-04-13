from django.db import models
from django.contrib.auth.models import AbstractUser, BaseUserManager
from django.contrib.auth.signals import user_logged_in
from django.utils.translation import gettext_lazy as _





class CustomUserManager(BaseUserManager):
    """Define a model manager for User model with no username field."""

    use_in_migrations = True

    def _create_user(self, email, password, **extra_fields):
        """Create and save a User with the given email and password."""
        if not email:
            raise ValueError('The given email must be set')
        user = self.model(email=email, **extra_fields)
        email = self.normalize_email(email)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_user(self, email, password=None, **extra_fields):
        """Create and save a regular User with the given email and password."""
        extra_fields.setdefault('is_staff', False)
        extra_fields.setdefault('is_superuser', False)
        return self._create_user(email, password, **extra_fields)

    def create_superuser(self, email, password, **extra_fields):
        """Create and save a SuperUser with the given email and password."""
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)

        if extra_fields.get('is_staff') is not True:
            raise ValueError('Superuser must have is_staff=True.')
        if extra_fields.get('is_superuser') is not True:
            raise ValueError('Superuser must have is_superuser=True.')

        return self._create_user(email, password, **extra_fields)


objects = CustomUserManager()

class CustomUser(AbstractUser):
    """拡張ユーザーモデル"""

    class Meta(object):
        db_table = 'custom_user'
    username = models.CharField(
        _('username'),
        max_length=150,
        blank=True,
        null=True,
        help_text="半角アルファベット、半角数字、@/./+/-/_ で150文字以下にしてください。",
        validators=[AbstractUser.username_validator],
        unique=True,
    )
    # メールアドレスを必須にしてユニーク制約を付与
    email = models.EmailField(_('email address'), unique=True, null=True, blank=True,)
    upload_count = models.IntegerField(verbose_name='アップロード回数', default=0)
    weight = models.FloatField(verbose_name='体重', null=True, blank=True, default=50)
    height = models.FloatField(verbose_name='身長', null=True, blank=True, default=160)

EMAIL_FIELD = 'email'
USERNAME_FIELD = 'username'
REQUIRED_FIELDS = ['username','email', 'height','weight']
