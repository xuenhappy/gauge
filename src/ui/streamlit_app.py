import os, sys, traceback, streamlit as st
PROJECT_ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),'../..'))
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
from src.eval.unified_loader import UnifiedQAPipeline
from src.models.registry import detect_run_method
st.set_page_config(page_title='Gauge-QA Demo', page_icon='🧭', layout='wide')
st.title('Gauge-QA Demo')
if 'pipeline' not in st.session_state: st.session_state.pipeline=None
if 'loaded_run_dir' not in st.session_state: st.session_state.loaded_run_dir=None
if 'loaded_method' not in st.session_state: st.session_state.loaded_method=None
with st.sidebar:
    run_dir=st.text_input('Run Directory', value='outputs/runs/gauge_qwen32b_squad_small_v1')
    device=st.selectbox('Device', ['cuda','cpu'], index=0)
    if st.button('加载模型'):
        try:
            with st.spinner('正在加载模型...'):
                method=detect_run_method(run_dir); pipe=UnifiedQAPipeline(run_dir, device=device)
            st.session_state.pipeline=pipe; st.session_state.loaded_run_dir=run_dir; st.session_state.loaded_method=method; st.success(f'加载成功：{method}')
        except Exception as e:
            st.error(f'加载失败：{e}'); st.code(traceback.format_exc())
context=st.text_area('Context', value='Paris is the capital of France.', height=180)
question=st.text_input('Question', value='What is the capital of France?')
if st.button('生成答案'):
    if st.session_state.pipeline is None: st.warning('请先加载模型。')
    else: st.write(st.session_state.pipeline.answer(context, question))
