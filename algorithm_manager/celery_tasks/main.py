from celery import Celery


# 创建celery应用
celery_app = Celery('algorithm')

# 导入celery配置
celery_app.config_from_object('celery_tasks.config')

# 导入任务
celery_app.autodiscover_tasks(['celery_tasks.train',
                               'celery_tasks.preprocess_task',
                               'celery_tasks.validate_task',
                               'celery_tasks.limit_task',
                               'celery_tasks.polyfit_task'])