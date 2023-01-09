import argparse
from datetime import datetime, timedelta
from tqdm import tqdm
from s3_utils import S3Client
import kss
import re
import time
def remove_special_char(corpus : list) -> list: #특수문자 제거
    return [' '.join(re.sub(r"""[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9!@#$%^&*()_+\-=\[\]\\{}|,.\/<>?;':"]"""," ",i).split()) for i in tqdm(corpus)]
def split_sentences(corpus_list):
    """ \n 기준으로 문장 나누기 """
    result = list()
    for i in tqdm(corpus_list) :
        result += i.split('\n')
    result = list(set(result))
    result.remove('')
    return result

def sentence_tokenize(data):
    """ sentence tokenizer를 통해 문장 나누기 """
    result = list()
    for i in tqdm(data) :
        result += kss.split_sentences(i)
    
    return result

def main(args):
    TODAY = datetime.now().strftime("%Y%m%d%H%M%S")
    FOLDER_NAME = 'spacing_corrector'
    
    #------------- Storage Client Connection----------#
    s3_client = S3Client(args.env)
    
    #-------------- Load Data Path -------------------#
    RAW_DATA_PREFIX = '{}/data/raw/'.format(FOLDER_NAME)
    PREPROCESSING_PREFIX = '{}/data/preprocessing/'.format(FOLDER_NAME)
    
    raw_data_key = "{}{}.pkl".format(RAW_DATA_PREFIX, args.corpus)
    raw_data = s3_client.get_pkl_data(raw_data_key)
    
    #--------------- Save Data End-point -------------#
    preprocessing_save_key = "{}preprocessing_{}.pkl".format(PREPROCESSING_PREFIX, TODAY)
    
    #-------------- Preprocessing data -------------------#
    preprocessing_data = split_sentences(raw_data)
    #이모지 제거, 특정 특수문자 및 아랍어 제거
    preprocessing_data = remove_special_char(preprocessing_data)
    
    #문장 분리
    # preprocessing_data = sentence_tokenize(preprocessing_data) #split 방법 1
    
    
    start = time.time()
    preprocessing_data = kss.split_sentences(preprocessing_data, num_workers=10) # split 방법 2
    preprocessing_data = list(itertools.chain(*preprocessing_data))
    end = time.time()
    s3_client.upload_data(preprocessing_save_key, preprocessing_data)


    sec = (end - start)
    result = timedelta(seconds=sec)
    result_list = str(timedelta(seconds=sec)).split(".")
    print('문장 split 수행 시간 :', result_list[0])
    
   
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='stage'
    )
    parser.add_argument(
        '--corpus',
        type=str,
        default='kowikitext'
    )
    args = parser.parse_args()
    main(args)
    