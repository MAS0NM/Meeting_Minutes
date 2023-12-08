from transformers import AutoTokenizer, AutoModel
import math

class chatGLM3:
    def __init__(self, tokenizer=None, model=None, model_path='default', text_processor=None, use_cuda=True):
        self.model_path = "THUDM/chatglm3-6b" if model_path == 'default' else model_path
        self.tokenizer = tokenizer if tokenizer else AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = model if model else AutoModel.from_pretrained(self.model_path, trust_remote_code=True).quantize(8).cuda()
        if use_cuda:
            self.model = self.model.cuda()
        self.model = self.model.eval()
        self.text_processor = text_processor
                    

    def chat(self, query):
        docs = self.text_processor.vector_store.similarity_search(query)
        prompt = f"Knowledge Known: \n{docs}\nAnswer Accordingly: \n{query}"
        resp = self.model.chat(self.tokenizer, prompt, history=[])
        print(resp)
        
                
    def generate_content(self):
        if not self.text_processor.para:
            content = self.text_processor.text
            assert content
            seg_size = 8192
            past_key_values, history = None, []
            current_length = 0
            prompt = self.text_processor.prompt
            self.model.stream_chat(self.tokenizer, prompt, history=history,
                                past_key_values=past_key_values,
                                return_past_key_values=True)
            
            for idx in range(len(math.ceil(self.text_processor.text)/8192)):
                input_text = self.text_processor.text[idx*seg_size: (idx+1)*seg_size]
                for response, history, past_key_values in self.model.stream_chat(self.tokenizer, input_text, history=history,
                                                                                past_key_values=past_key_values,
                                                                                return_past_key_values=True):
                    yield response[current_length:]
                    current_length = len(response)
        else:
            content = self.text_processor.para
            prompt = self.text_processor.prompt
            current_length = 0
            past_key_values, history = None, []
            self.model.stream_chat(self.tokenizer, prompt, history=history,
                                past_key_values=past_key_values,
                                return_past_key_values=True)
            for idx, text in enumerate(self.text_processor.para):
                if not text:
                    continue
                print(f'------------------{idx}------------------')
                print(text)
                for response, history, past_key_values in self.model.stream_chat(self.tokenizer, text, history=history,
                                                                                past_key_values=past_key_values,
                                                                                return_past_key_values=True):
                    yield response[current_length:]
                    current_length = len(response)