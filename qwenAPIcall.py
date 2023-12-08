from http import HTTPStatus
import dashscope
import math

class qw:
    def __init__(self, text_processor=None):
        self.text_processor = text_processor
        
    
    def call_with_prompt(self, query):
        dashscope.api_key = 'sk-2f4c3024b43d4d0c98672ddb02096f13' # 将 YOUR_API_KEY 改成您创建的 API-KEY
        response = dashscope.Generation.call(
            model=dashscope.Generation.Models.qwen_turbo,
            prompt=self.text_processor.prompt + '\n' + query
        )
        if response.status_code == HTTPStatus.OK:
            return response.output['text']
        else:
            print(response.code)  # The error code.
            print(response.message)  # The error message.
                
    def generate_content(self):
        if not self.text_processor.para:
            content = self.text_processor.text
            assert content
            seg_size = 8192
            
            for idx in range(len(math.ceil(self.text_processor.text)/seg_size)):
                input_text = self.text_processor.text[idx*seg_size: (idx+1)*seg_size]
                response = self.call_with_prompt(input_text)
                yield response
        else:
            content = self.text_processor.para
            for idx, input_text in enumerate(self.text_processor.para):
                if not input_text:
                    continue
                print(f'------------------{idx}------------------')
                print(input_text)
                response = self.call_with_prompt(input_text)
                yield response, ''