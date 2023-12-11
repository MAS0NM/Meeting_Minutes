## Toy Project

Constructed mainly with funasr, langchain, modelscope and streamlit

llm is not inferencing locally, call with your own api key

`streamlit run ui.py` to run

更换asr模型参考fun_asr.py文件，使用fun_asr已集成的模型，更改pipline内的参数；使用fun_asr以外的模型和框架，重起文件实现同名函数

更换llm api参考sparkAPIcall.py和qwenAPIcall.py，重起文件实现同名函数