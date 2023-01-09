import argparse
from datetime import datetime
from Korpora import Korpora
from s3_utils import S3Client
import sys
from io import StringIO

def main(args):
    s3_client = S3Client(args.env)
    FOLDER_NAME = "spacing_corrector"
    
    RAW_DATA_PREFIX = "{}/data/raw/".format(FOLDER_NAME)
    raw_data_save_key = "{}{}.pkl".format(RAW_DATA_PREFIX, args.corpus)
    

    sys.stdin = StringIO('yes')    
    data = Korpora.load(args.corpus)
    print('Succeed load data')
    data = data.get_all_texts()
    


    s3_client.upload_data(raw_data_save_key, data)
    print('Succeed upload')
    
    
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='stage',
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='kowikitext'
    )
    args=parser.parse_args()
    main(args)