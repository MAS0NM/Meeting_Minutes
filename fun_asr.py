# from funasr import infer
import torchaudio
import librosa
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class asr:
    def __init__(self):
        self.asr_model = pipeline(
                    task=Tasks.auto_speech_recognition,
                    model='damo/speech_paraformer-large-vad-punc-spk_asr_nat-zh-cn',
                    model_revision='v0.0.2',
                    vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
                    punc_model='damo/punc_ct-transformer_cn-en-common-vocab471067-large',
                    output_dir='./',
                )
        self.wav = []
        self.wav_length = 0
        self.cur_file = ''
        self.asr_res = []
        self.asr_res_with_time_spk = []
        self.sr = 16000
        
        
    def load_wav(self, file_path, sr=16000):
        self.sr = sr
        if os.path.isfile(file_path):
            speech, _ = librosa.load(file_path, sr=sr)
        else:
            speech = file_path
        self.wav = speech
        self.cur_file = file_path
        speech_length = speech.shape[0]
        print(speech_length)
        self.wav_length = speech_length
        
        
    def get_time(self, num):
        total_seconds = num // 1000
        h = total_seconds // 3600
        m = total_seconds % 3600 // 60
        s = total_seconds % 60
        m = str(m) if m > 9 else '0' + str(m)
        s = str(s) if s > 9 else '0' + str(s)
        return f'{h}:{m}:{s}' if h else f'{m}:{s}'
    
    
    def hms2sec(self, hms):
        time_part = hms.split(':')
        if len(time_part) == 3:
            h, m, s = map(int, time_part)
            return h*3600+m*60+s 
        elif len(time_part) == 2:
            m, s = map(int, time_part)
            return m*60+s
    
          
    def time_interval(self, hms1, hms2):
        return self.hms2sec(hms1) - self.hms2sec(hms2) 
        
        
    def wav2text(self, file_path=''):
        if len(file_path) and file_path != self.cur_file:
            self.cur_file = file_path
        res = self.asr_model(audio_in=self.cur_file, batch_size_token=5000, batch_size_token_threshold_s=40, max_single_segment_time=6000)
        res = [{'sentence': d['text'], 'start': self.get_time(d['start']), 'end': self.get_time(d['end']), 'spk': d['spk']} for d in res['sentences']]
        self.asr_res = res
        return res


    def dump(self, output_file_path):
        json_str = json.dumps(self.asr_res, ensure_ascii=False, cls=NumpyEncoder)
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(json_str)
            
            
    def get_asr_res(self):
        # res = ''.join([dic['sentence'] for dic in self.asr_res])
        res = []
        line = ''
        previous_spk = '00:00'
        previous_time_end = '00:00'
        for dic in self.asr_res:
            spk_id = dic['spk']
            start_time = dic['start']
            sent = dic['sentence']
            if line == '':
                line = f'{start_time}: speaker_{spk_id}: {sent}'
                previous_spk = spk_id
                previous_time_end = dic['end']
            elif spk_id != previous_spk or len(line) > 500 and self.time_interval(start_time, previous_time_end) <= 3:
                res.append(line+'\n')
                line = f'{start_time}: speaker_{spk_id}: {sent}'
                previous_spk = spk_id
                previous_time_end = dic['end']
            elif self.time_interval(start_time, previous_time_end) > 3:
                line += '\n' + sent
                previous_spk = spk_id
                previous_time_end = dic['end']
            else:
                line += sent
        res.append(line+'\n')
        self.asr_res_with_time_spk = res
        with open('./meeting_record.txt', 'w', encoding='utf8') as f:
            for line in self.asr_res_with_time_spk:
                f.write(line)
        return res
        
# logger = logging.getLogger('modelscope')
# logger.setLevel(logging.ERROR)
# myasr = asr()
# myasr.load_wav('男声_3.wav')
# print(myasr.wav2text())
# for res in myasr.inf():
#     print(res, end='', flush=True)
