from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import time
import os
import nltk
from nltk.corpus import stopwords
import jieba
# nltk.download('stopwords')
stopwords_cn_list = stopwords.words('chinese')

class TextChain:
    def __init__(self):
        self.prompt = ''
        self.structured_text = []
        self.embed_method = None
        self.vector_store = None
        
        
    def init_embed_method(self):
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        self.embed_method = HuggingFaceBgeEmbeddings(
                        model_name=model_name,
                        model_kwargs=model_kwargs,
                        encode_kwargs=encode_kwargs
                    )
        
        
    def set_prompt(self, mode='meetingminutes', prompt=''):
        if 'default' in mode or 'meetingminutes' in mode and prompt=='':
            prompt = '请用简洁的语言对后续文本进行概括，改进它的逻辑链并修正可能存在的表述错误，但要保证原意并且不能引入额外的任何信息，要概括的要点如下：\
            ###会议主题\n###\
            ###会议中提出的问题，并跟随其解决方案\
            ###如有时间、地点、人物名称事务名称和产品名称，请务必强调'
            self.prompt = prompt
        else:
            self.prompt = prompt
            
    
    def structured_load(self, struct_list):
        if type(struct_list) == str and os.path.isfile(struct_list):
            with open(struct_list, 'r', encoding='utf-8') as f:
                struct_list = json.load(f)
                # print(struct_list)
        self.structured_text = struct_list
        # print(self.structured_text)
        
        
    def sentence_embed(self):
        if not self.embed_method:
            self.init_embed_method()
        self.vector_store = []
        for dic in self.structured_text:
            sentence = dic['sentence']
            vec = self.embed_method.embed_query(sentence)
            dic['vec'] = np.array(vec)
            self.vector_store.append(vec)
        self.vector_store = np.array(self.vector_store)
        
        
    def find_threshold_on_k(self, k, sim_list):
        tmp = sim_list - k
        print(tmp.shape)
        return np.argmin(np.abs(tmp)) / len(tmp)
    
    
    def chunk_all(self, chunk_size=4096, remove_stop_words=False):
        '''
            segment the whole context into chunks which in the length of around the designated `chunk_size`
            by default in the following step it will remove the stopwords, set False to `remove_stop_word` to skip
            return format would be a list full of strings
        '''
        fulltext = ''.join([dic['sentence'] for dic in self.structured_text])
        text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_size//2, separator='。')
        fulltext = text_splitter.split_text(fulltext)
        if remove_stop_words:
            fulltext_without_stopwords = []
            for sent in fulltext:
                sent = ''.join([word for word in jieba.cut(sent, use_paddle=True) if word not in stopwords_cn_list])
                fulltext_without_stopwords.append(sent)
            fulltext = fulltext_without_stopwords
        self.para = fulltext
        
        
    def paragraphing(self, k=3):
        '''
            embed all sentences, calculate for a similarity co-ocurrence matrix, reparagrahing based on that
            sentence embeding is time cosuming, not suggested to use this function
        '''
        if self.vector_store is None or len(self.vector_store) == 0:
            self.sentence_embed()
        similarity_matrix = []
        similarity_matrix = cosine_similarity(self.vector_store)
        similarity_matrix = np.array([1]+[similarity_matrix[i+1][i] for i in range(len(similarity_matrix)-1)])
        time1 = time.time()
        sim_list = np.array([np.sum(len(np.where(similarity_matrix <= s/1000)[0])) for s in range(1000)])
        print(time.time() - time1)
        # threshold = sum(similarity_matrix) / len(similarity_matrix)
        threshold = self.find_threshold_on_k(k, sim_list)
        para = []
        time_stamp = []
        cur_chunk = ''
        cur_start = 0
        cur_end = 0
        for i in range(len(self.structured_text)):
            if similarity_matrix[i] >= threshold:
                cur_chunk += self.structured_text[i]['sentence']
            else:
                if cur_chunk:
                    para.append(cur_chunk)
                    time_stamp.append([cur_start, cur_end])
                cur_chunk = self.structured_text[i]['sentence']
                cur_start = self.structured_text[i]['start']
                cur_end = self.structured_text[i]['end']
        if cur_chunk:
            para.append(cur_chunk)
            time_stamp.append([cur_start, cur_end])
        self.para = para
        self.time_stamp = time_stamp
        
            
# if __name__ == "__main__":
#     # myasr = asr()
#     # myasr.load_wav('Eval_Ali\R8001_M8004_MS801.wav')
#     # myasr.load_wav('男声_3.wav')
#     # structured_text = myasr.wav2text()
#     # myasr.dump('./asr_res.json')
#     with open('./asr_res.json', 'r', encoding='utf-8') as f:
#         structured_text = json.load(f)
#     print(structured_text)
#     tc = TextChain()
#     tc.structured_load(structured_text)
#     # tc.paragraphing()
#     tc.chunk_all()
    