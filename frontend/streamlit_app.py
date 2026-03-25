import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

st.set_page_config(page_title='Enterprise Agentic RAG', layout='wide')
st.title('Enterprise Document Q&A with Agentic RAG')
st.caption('Upload enterprise documents, index them, and ask grounded questions.')

with st.sidebar:
    st.header('Upload Documents')
    uploaded_files = st.file_uploader(
        'Upload PDF, TXT, CSV, XLSX, or XLS files',
        type=['pdf', 'txt', 'csv', 'xlsx', 'xls'],
        accept_multiple_files=True,
    )

    if st.button('Ingest Documents', use_container_width=True):
        if not uploaded_files:
            st.warning('Please upload at least one document.')
        else:
            files = [('files', (f.name, f.getvalue(), f.type or 'application/octet-stream')) for f in uploaded_files]
            with st.spinner('Indexing documents...'):
                response = requests.post(f'{API_BASE_URL}/api/v1/ingest', files=files, timeout=180)
            if response.ok:
                data = response.json()
                st.success(f"Processed {data['files_processed']} file(s) and created {data['chunks_created']} chunk(s).")
                st.write('Sources:', ', '.join(data['sources']))
            else:
                st.error(response.text)

st.subheader('Ask a Question')
question = st.text_area('Enter a question about your uploaded documents', height=120)
top_k = st.slider('Top-K retrieved chunks', min_value=1, max_value=8, value=4)

if st.button('Ask', type='primary'):
    if not question.strip():
        st.warning('Please enter a question.')
    else:
        payload = {'question': question, 'top_k': top_k}
        with st.spinner('Thinking with agents...'):
            response = requests.post(f'{API_BASE_URL}/api/v1/ask', json=payload, timeout=180)
        if response.ok:
            result = response.json()
            st.markdown('### Answer')
            st.write(result['answer'])

            with st.expander('Citations', expanded=True):
                for citation in result['citations']:
                    st.markdown(f"**{citation['source']}** — `{citation['chunk_id']}`")
                    st.write(citation['preview'])

            with st.expander('Agent Trace'):
                for step in result['agent_trace']:
                    st.write(f'- {step}')
        else:
            st.error(response.text)
