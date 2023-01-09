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
        
    def get_pkl_data(self, data_key) :
        s3 = boto3.client('s3',
                          aws_access_key_id = self.ACCESSKEY,
                          aws_secret_access_key = self.SECRETKEY)
        s3_object = s3.get_object(Bucket=self.BUCKET_NAME, Key=data_key)['Body']
        byte_data = s3_object.read()
        data = BytesIO(byte_data)

        return pickle.load(data)
    
    def get_data(self, data_key) :
        s3 = boto3.client('s3',
                          aws_access_key_id = self.ACCESSKEY,
                          aws_secret_access_key = self.SECRETKEY)
        s3_object = s3.get_object(Bucket=self.BUCKET_NAME, Key=data_key)['Body']
        byte_data = s3_object.read()
        data = byte_data.decode('utf-8')

        return data
        
    
    def upload_data(self, save_key, data):
        s3 = boto3.client('s3',
                          aws_access_key_id = self.ACCESSKEY,
                          aws_secret_access_key = self.SECRETKEY,
                          )
        byte_io = BytesIO()
        pickle.dump(data, byte_io)
        s3.put_object(Bucket = self.BUCKET_NAME, Body = byte_io.getvalue(), Key = save_key)
    
    def get_latest_file_path(self, storage_path):
        s3 = boto3.client('s3',
                          aws_access_key_id = self.ACCESSKEY,
                          aws_secret_access_key = self.SECRETKEY,
                          )

        contents_list = s3.list_objects(Bucket=self.BUCKET_NAME, Prefix=storage_path)['Contents']
        storage_path = [i['Key'] for i in contents_list][-1]

        file_name = storage_path.split('/')[-1]
        print(storage_path)
        return storage_path
    
    def get_data(self, data_key) :
        s3 = boto3.client('s3',
                          aws_access_key_id = self.ACCESSKEY,
                          aws_secret_access_key = self.SECRETKEY)
        s3_object = s3.get_object(Bucket=self.BUCKET_NAME, Key=data_key)['Body']
        byte_data = s3_object.read()
        data = byte_data.decode('utf-8')

        return data
        
    