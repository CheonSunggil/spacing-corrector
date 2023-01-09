from s3_config import LiveConfig, StageConfig
from io import BytesIO
import boto3
import pickle
class S3Client(object):
    def __init__(self, env):
        if env == 'stage' :
            config = StageConfig()
            
        elif env == 'local' :
            config = StageConfig()
        else :
            config = LiveConfig()
        
        self.BUCKET_NAME = config.S3_BUCKET
        self.ACCESSKEY = config.S3_ACCESS_KEY
        self.SECRETKEY = config.S3_SECRET_KEY
    
    
    def upload_data(self, save_key, data):
        s3 = boto3.client('s3',
                          aws_access_key_id = self.ACCESSKEY,
                          aws_secret_access_key = self.SECRETKEY,
                          )
        byte_io = BytesIO()
        pickle.dump(data, byte_io)
        s3.put_object(Bucket = self.BUCKET_NAME, Body = byte_io.getvalue(), Key = save_key)
        
    