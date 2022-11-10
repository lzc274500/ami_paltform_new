# gunicorn.conf
# coding:utf-8
import multiprocessing

broker_url = "redis://127.0.0.1/1"
worker_concurrency = multiprocessing.cpu_count() // 2
worker_max_tasks_per_child = 1