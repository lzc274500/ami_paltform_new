# gunicorn.conf
# coding:utf-8
import multiprocessing

# 绑定的ip与端口
bind = '0.0.0.0:8880'
preload_app = True
# 并行工作进程数, int，cpu数量*2+1 推荐进程数
# workers = multiprocessing.cpu_count() * 2 + 1
workers = multiprocessing.cpu_count()
# 指定每个进程开启的线程数
# threads = 2
# 设置守护进程,将进程交给supervisor管理
daemon = False
# 工作模式协程，默认的是sync模式
worker_class = 'egg:meinheld#gunicorn_worker'
# 设置最大并发量（每个worker处理请求的工作线程数，正整数，默认为1）
worker_connections = 1000
# 最大客户端并发数量，默认情况下这个值为1000。此设置将影响gevent和eventlet工作模式
max_requests = 1000
# 设置进程文件目录
# pidfile = '/home/your_path/gunicorn.pid'
# 设置访问日志和错误信息日志路径
logconfig_dict = {
    'version':1,
    'disable_existing_loggers': False,
    "root": {
      "level": "INFO",
      "handlers": ["console"]
        },
    'loggers':{
        "gunicorn.error": {
            "level": "INFO",# 打日志的等级可以换的，下面的同理
            "handlers": ["error_file"], # 对应下面的键
            "propagate": 1,
            "qualname": "gunicorn.error"
        },

        "gunicorn.access": {
            "level": "DEBUG",
            "handlers": ["access_file"],
            "propagate": 0,
            "qualname": "gunicorn.access"
        }
    },
    'handlers':{
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 1024*1024*1024,# 打日志的大小，我这种写法是1个G
            "backupCount": 1,# 备份多少份，经过测试，最少也要写1，不然控制不住大小
            "formatter": "generic",# 对应下面的键
            # 'mode': 'w+',
            "filename": "/home/algorithm_manager/log/error.log"# 打日志的路径
        },
        "access_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "maxBytes": 1024*1024*1024,
            "backupCount": 1,
            "formatter": "generic",
            "filename": "/home/algorithm_manager/log/access.log",
        },
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'generic',
        },
    },
    'formatters':{
        "generic": {
            "format": "'[%(process)d] [%(asctime)s] %(levelname)s [%(filename)s:%(lineno)s] %(message)s'", # 打日志的格式
            "datefmt": "[%Y-%m-%d %H:%M:%S %z]",# 时间显示方法
            "class": "logging.Formatter"
        }
    }
}

# accesslog = '/home/algorithm_manager/log/access.log'
# errorlog = 'E:/PycharmProjects/nodiot-algorithm/algorithm_manager/log/error.log'
# errorlog = '/home/algorithm_manager/log/error.log'
# 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
# loglevel = 'info'