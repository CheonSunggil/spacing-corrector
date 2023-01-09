class LocalConfig():
    def __init__(self):
        # AWS S3
        self.S3_BUCKET = 'sg-project-live'
        self.S3_ACCESS_KEY = 'AKIA4SLFIUDHF7U4KS2O'
        self.S3_SECRET_KEY = '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'
        self.APM_URL = ''
        self.APM_USE = False
        self.KAFKA_USE = False
        self.BOOTSTRAP_SERVERS = ''

class LiveConfig():
    def __init__(self):
        # AWS S3
        self.S3_BUCKET = 'sg-project-live'
        self.S3_ACCESS_KEY = 'AKIA4SLFIUDHF7U4KS2O'
        self.S3_SECRET_KEY = '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'
        self.APM_URL = '',
        self.APM_USE = True,
        self.KAFKA_USE = True
        self.BOOTSTRAP_SERVERS = ''


class StageConfig():
    def __init__(self):
        # AWS S3
        self.S3_BUCKET = 'sg-project-live'
        self.S3_ACCESS_KEY = 'AKIA4SLFIUDHF7U4KS2O'
        self.S3_SECRET_KEY = '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'
        self.APM_URL = ''
        self.APM_USE = False
        self.KAFKA_USE = True
        self.BOOTSTRAP_SERVERS = ''