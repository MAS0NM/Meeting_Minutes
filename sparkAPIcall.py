from spark_gpt.spark_gpt import SparkGPT
import math

class spk:
    def __init__(self, text_processor=None):
        self.text_processor = text_processor
        self.model = SparkGPT(self.text_processor.prompt)
                
    def generate_content(self):
        if not self.text_processor.para:
            content = self.text_processor.text
            assert content
            seg_size = 8192
            current_length = 0
            
            for idx in range(len(math.ceil(self.text_processor.text)/seg_size)):
                input_text = self.text_processor.text[idx*seg_size: (idx+1)*seg_size]
                response = self.model.ask(input_text)
                yield response[current_length:]
        else:
            content = self.text_processor.para
            for idx, input_text in enumerate(self.text_processor.para):
                if not input_text:
                    continue
                print(f'------------------{idx}------------------')
                print(input_text)
                response = self.model.ask(input_text)
                # if self.text_processor.time_stamp:
                #     yield response, self.text_processor.time_stamp[idx][0]
                # else:
                #     yield response, ''
                yield response, ''
                    