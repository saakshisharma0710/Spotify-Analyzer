import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os

from audio_processor import AudioProcessor


class PredictionInterface:
    def __init__(self, data_analyzer):
        self.analyzer = data_analyzer
    
    def show_prediction_dashboard(self):
        st.markdown('<h2 class="section-header">🤖 AI Hit Prediction</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            self._show_audio_upload_section()
        
        with col2:
            self._show_prediction_output()
    
    def _show_audio_upload_section(self):
        st.markdown("### 🎵 Upload Audio for Prediction")
        
        uploaded_file = st.file_uploader(
            "Upload 30-second audio clip",
            type=['wav', 'mp3', 'm4a']
        )
        
        if uploaded_file:
            st.audio(uploaded_file)
            st.session_state['uploaded_audio'] = uploaded_file
            st.success("✅ Audio ready!")
            
            if st.button("🚀 Process with ML Model"):
                processor = AudioProcessor()
                features = processor.extract_features(uploaded_file)
                st.session_state['extracted_features'] = features
                
                # model returns % directly (0–100)
                prob = model_predict.predict(features)
                st.session_state['prediction_prob'] = prob
                st.session_state['prediction_done'] = True
    
    def _show_prediction_output(self):
        st.markdown("### 📊 Prediction Output")
        
        if st.session_state.get('prediction_done'):
            
            prob = st.session_state['prediction_prob']
            features = st.session_state['extracted_features']
            
            st.success(f"🎯 Hit Probability: **{prob:.2f}%**")
            
            st.subheader("Extracted Feature Summary")
            df = pd.DataFrame(features.items(), columns=["Feature", "Value"])
            st.dataframe(df)
        
        else:
            st.info("Upload a file & click process to see results.")
