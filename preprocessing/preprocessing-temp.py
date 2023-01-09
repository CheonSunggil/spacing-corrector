from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

def space_tagging(corpus_list):
    """ character 단위 도큰화와 띄어쓰기 태깅 수행 
    """
    words_list = list()
    tags_list = list()
    for corpus in tqdm(corpus_list) :
        sentence = corpus.split(" ")
        tags = list()
        words = list()
        for word in sentence :
            for i,w in enumerate(word) :
                words.append(w)
                if i == 0:
                    tags.append('B')
                else :
                    tags.append('I')
        words_list.append(words)
        tags_list.append(tags)
  
    return (words_list, tags_list)

def train_tokenizer(self, sentences, tags):
        """ 토크나이저 학습
        """
        
        self.sent_tokenizer = Tokenizer(oov_token='OOV', lower=False)
        self.tag_tokenizer = Tokenizer(lower=False)
        
        self.sent_tokenizer.fit_on_texts(sentences)
        self.tag_tokenizer.fit_on_texts(tags)
        
        self.sent_tokenizer.fit_on_texts(create_all_hangul_char()) #모든 한글 단어 토큰 추가
        
        return self.sent_tokenizer, self.tag_tokenizer 