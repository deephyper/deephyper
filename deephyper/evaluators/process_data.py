from balsam.launcher.dag import BalsamJob

data = BalsamJob.objects.values_list('data__reward', flat=True)

print(f'data: {data}')
