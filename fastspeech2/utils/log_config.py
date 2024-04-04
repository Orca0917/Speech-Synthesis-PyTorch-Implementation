import logging.config

def setup_logging():
    logging.config.dictConfig({
        'version': 1,
        'formatters': {'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        }},
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'level': 'INFO'
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.FileHandler',
                'filename': 'logs/ex.log',
                'formatter': 'default',
            },
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
        },
    })


setup_logging()