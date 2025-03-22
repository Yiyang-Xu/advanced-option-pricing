import numpy as np
import pandas as pd
import os
from src.ui import app
import streamlit as st

def load_css():
    """Load CSS styles for the application"""
    css_path = os.path.join("static", "styles.css")  # 指定正确路径
    if os.path.exists(css_path):
        with open(css_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ styles.css File Not Found, Please Check The Path")

if __name__ == "__main__":
    load_css()
    app()