'''
    deprecated
    ！！！！！！！
'''


import math
import langchain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema.document import Document
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from utils import preprocess, IoU
import numpy as np

class TextProcessor:
    def __init__(self, tokenizer=None, model=None, model_path='default', use_cuda=True):
        self.prompt = ''
        self.text = []
        self.docs = None
        self.vector_store = None
        self.bow = None
        self.chunk_size = 200
        self.overlap = 50
        self.para = []
        self.structured_text = []
        
        
    def set_prompt(self, mode='meetingminutes'):
        if 'default' in mode or 'meetingminutes' in mode:
            prompt = '请用简洁的语言对后续文本进行概括，改进它的逻辑链并修正可能存在的表述错误，但要保证原意并且不能引入额外的任何信息，要概括的要点如下：\
            ###会议主题\n###\
            ###会议中提出的问题，并跟随其解决方案\
            ###如有时间、地点、人物名称事务名称和产品名称，请务必强调'
            self.prompt = prompt
            
            
    def structured_load(self, struct_list):
        self.structured_text = struct_list
                        
    
    def langchain_load(self, file_path):
        '''
            load the doc according to its path
            vertorize and store
        '''
        loader = UnstructuredFileLoader(file_path, mode='single')
        docs = loader.load()
        self.docs = docs
        full_content = [doc.page_content for doc in docs]
        full_content = ''.join(full_content)
        print(full_content)
        text_splitter = CharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        full_text = text_splitter.split_text(full_content)
        self.text = full_text
        print(len(self.text))
        print(self.text)
        # model_name = "BAAI/bge-large-en"
        # model_kwargs = {'device': 'cpu'}
        # encode_kwargs = {'normalize_embeddings': True}
        # embeddings = HuggingFaceBgeEmbeddings(
        #                 model_name=model_name,
        #                 model_kwargs=model_kwargs,
        #                 encode_kwargs=encode_kwargs
        #             )
        # docs = [Document(page_content=doc, metadata={"index": i}) for i, doc in enumerate(self.text)]
        # print(f'len docs: {len(docs)}')
        # vector_store = FAISS.from_documents(docs, embeddings)
        # self.vector_store = vector_store
        # self.make_bow()
        
        
    def make_bow(self):
        context = preprocess(self.text)
        dictionary = Dictionary(context)
        corpus = [dictionary.doc2bow(chunk) for chunk in context]
        self.bow_in_chunks = corpus
        self.dictionary = dictionary
        
        
    def topic_modeling(self):
        dictionary = self.dictionary
        corpus = self.bow_in_chunks
        lda = LdaModel(
            corpus, 
            num_topics=10,
            id2word=dictionary,
            )
        # print(lda.top_topics(corpus))
        top_top = []
        for chunk in self.bow_in_chunks:
            top_top.append([x[1] for x in lda.top_topics([chunk])[0][0]])
        return top_top
               
        
    def calculate_similarity_between_blocks(self, mode='topic'):
        similarity_results = []
        
        if mode == 'sentence':
            context = self.text
            vector = self.vector_store
            for i in range(len(context)):
                query = context[i]
                results_with_scores = vector.similarity_search_with_score(query, k=len(context))
                rws = []
                for doc, score in results_with_scores:
                    # print(f"Content: {doc.page_content}, Metadata: {doc.metadata}, Score: {score}")
                    rws.append([doc.metadata['index'], score])
                rws.sort(key=lambda x: x[0])
                similarity_results.append([x[1] for x in rws])
            new = []
            for i in range(len(similarity_results)-1):
                new.append(similarity_results[i+1][i])
            similarity_results = new
            return np.array(similarity_results)
        
        elif mode == 'topic':
            tops = self.topic_modeling()
            IoUs = []
            for i in range(len(tops)-1):
                IoUs.append(IoU(tops[i], tops[i-1]))
            return np.array(IoUs)
                
        
    def paragraphing(self):
        print('start paragraphing')
        sims = self.calculate_similarity_between_blocks()
        threshold = sum(sims) / len(sims)
        para = []
        cur_chunk = ''
        for i in range(len(self.text)-1):
            if sims[i] > threshold:
                cur_chunk = cur_chunk[:-self.overlap] + self.text[i+1]
            else: 
                if cur_chunk != '':
                    para.append(cur_chunk)
                cur_chunk = self.text[i]
        self.para = para
        print('paragraphing compeleted')
        print(para)
        return para
            
        
    def chat(self, query):
        docs = self.vector_store.similarity_search(query)
        prompt = f"Knowledge Known: \n{docs}\nAnswer Accordingly: \n{query}"
        resp = self.model.chat(self.tokenizer, prompt, history=[])
        print(resp)
        
                
    def generate_content(self):
        if not self.para:
            content = self.text
            assert content
            seg_size = 8192
            past_key_values, history = None, []
            current_length = 0
            prompt = self.prompt
            self.model.stream_chat(self.tokenizer, prompt, history=history,
                                past_key_values=past_key_values,
                                return_past_key_values=True)
            
            for idx in range(len(math.ceil(self.text)/8192)):
                input_text = self.text[idx*seg_size: (idx+1)*seg_size]
                for response, history, past_key_values in self.model.stream_chat(self.tokenizer, input_text, history=history,
                                                                                past_key_values=past_key_values,
                                                                                return_past_key_values=True):
                    yield response[current_length:]
                    current_length = len(response)
        else:
            content = self.para
            prompt = self.prompt
            current_length = 0
            past_key_values, history = None, []
            self.model.stream_chat(self.tokenizer, prompt, history=history,
                                past_key_values=past_key_values,
                                return_past_key_values=True)
            for idx, text in enumerate(self.para):
                if not text:
                    continue
                print(f'------------------{idx}------------------')
                print(text)
                for response, history, past_key_values in self.model.stream_chat(self.tokenizer, text, history=history,
                                                                                past_key_values=past_key_values,
                                                                                return_past_key_values=True):
                    yield response[current_length:]
                    current_length = len(response)
                    
                
if __name__ == '__main__':
    chatchat = TextProcessor()
    chatchat.langchain_load('test.txt')
    # chatchat.make_bow()
    # print(chatchat.calculate_similarity_between_blocks())
    # print(chatchat.paragraphing())
    # chatchat.chat('What is this file mainly about?')