import streamlit as st
import librosa
import time
import soundfile as sf

st.markdown('MEETING MINUTES')
uploaded_file = st.file_uploader('Drag .wav file here', type=['wav', 'mp3'])
# log_info = st.text_area('log')

st.session_state['generated'] = []
st.session_state['state'] = 'waiting'
st.session_state['asr'] = ''

# wav, sr = librosa.load('Eval_Ali\R8008_M8013_N_SPK8047.wav', sr=16000)
# sf.write('Eval_Ali\R8008_M8013_N_SPK8047.wav', wav, sr)


display_content = ''
if "text" not in st.session_state:
    st.session_state.text_content = display_content
title_asr = st.empty()
asr_content = st.empty()
timer_asr = st.empty()
# generate_content_area = st.text_area('Generated Content', value=st.session_state["text_content"], key="generated_content")
title_gc = st.empty()
mm_content = st.empty()
timer_llm = st.empty()

import workflow

if uploaded_file is not None:
    if st.session_state['state'] == 'done':
        mm_content.empty()
    audio, sr = librosa.load(uploaded_file, sr=16000)
    st.session_state['state'] = 'generating'
    
    start_time = time.time()
    asr_res = workflow.wav2txt(audio)
    title_asr.write('ASR RESULT')
    asr_content.write(asr_res)
    timer_asr.write(str(int(time.time()-start_time))+'s')
      
    start_time = time.time()
    # print(uploaded_file)
    for content, timestamp in workflow.stream_gen()():
        # display_content += result
        st.session_state['generated'].append(content)
        title_gc.write('GENERATED CONTENT')
        # mm_content.write(timestamp)
        mm_content.write(''.join(st.session_state['generated']))
        timer_llm.write(str(int(time.time()-start_time))+'s')
        
    st.session_state['state'] = 'done'
    
if st.session_state['state'] == 'done':
    st.session_state['generated'] = []
    