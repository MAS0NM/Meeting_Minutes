# from audio_process import wspr
from textchain import TextChain
# from chatGLM import chatGLM3
from sparkAPIcall import spk
from qwenAPIcall import qw

import os
import librosa
from fun_asr import asr
# from spark_gpt.spark_gpt import SparkGPT


# wspr_model = wspr()
asr_model = asr()
tc = TextChain()
tc.set_prompt()
# llm = spk(text_processor=tc)
llm = qw(text_processor=tc)

def wav2txt(wav, sr=16000):
    asr_model.load_wav(wav, sr)
    asr_model.wav2text()
    asr_model.dump('asr_res.json')
    return asr_model.get_asr_res()


# def gen(wav_path='男声_3.wav'):
#     wav2txt(wav_path)
#     yield from abstract()


def abstract(file_path='./asr_res.json'):
    tc.structured_load(file_path)
    tc.chunk_all()
    tc.set_prompt()
    return llm.generate_content()
    
    
def stream_gen(func=abstract):
    '''
        only if the llm call function support stream generation
    '''
    def wrapper(*args, **kwargs):
        for content, timestamp in func(*args, **kwargs):
            yield content, timestamp
    return wrapper
        
# audio, sr = librosa.load('男声_3.wav', sr=16000)

# for result in stream_gen(audio)():
#     print(result, end='', flush=True)
