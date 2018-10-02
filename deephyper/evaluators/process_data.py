"""
https://docs.djangoproject.com/en/2.1/ref/contrib/postgres/fields/#django.contrib.postgres.fields.JSONField
"""

from balsam.launcher.dag import BalsamJob


res = BalsamJob.objects.filter(workflow='test')
res = res.values_list('data', flat=True)

print(res)

