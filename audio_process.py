import whisper
import librosa
import numpy as np
import math

class wspr:
    def __init__(self, wspr_model=None):
        self.wspr = self.init_whisper(model='small') if not wspr_model else wspr_model
        self.slices = []
        self.text_pool = ''
        self.tmp_wave = None
        self.sr = 16000
        
    def init_whisper(self, model='small'):
        wspr = whisper.load_model(model)
        return wspr
    
    def load_audio(self, wave, sr=16000):
        '''
            the wave here could be a file path 
            or a np.array that is already loaded from methods like librosa.load()
        '''
        if isinstance(wave, str):
            wave, sr = librosa.load(wave, sr=sr)
        self.tmp_wave = wave
        self.sr = sr

    def speech2text(self):
        '''
            convert the already loaded wave into text content
            
        '''
        assert len(self.tmp_wave)
        if self.slices:
            for sl in self.slices:
                res = self.wspr.transcribe(sl)
                self.text_pool += res['text'] + '\n'
        else:
            res = self.wspr.transcribe(self.tmp_wave)
            self.text_pool
            
    def write_text_file(self, write_file_path='ASR_result.txt'):
        with open(write_file_path, 'w', encoding='utf8') as f:
            f.write(self.text_pool)
    
    def get_asr_res(self):
        return self.text_pool
        
    def wav_slice(self, slice_sec=10, use_overlap_window=True):
        '''
            slice the loaded wave with a window of 10 seconds
            stored in self.slices in the shape below: 
            [[slice_length], [slice_length], ..., [residual without padding]]
        '''
        slice_length = slice_sec * self.sr
        assert len(self.tmp_wave) != 0
        L = len(self.tmp_wave)
        bias = slice_length // 5000 if use_overlap_window else 0
        for idx in range(math.ceil(L/slice_length)):
            start = idx*slice_length if idx == 0 else idx*slice_length - bias
            end = (idx+1)*slice_length
            self.slices.append(self.tmp_wave[start: end])
            
    