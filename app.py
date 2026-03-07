import os
import streamlit as st
from db import create_tables
import ui as ui

st.set_page_config(
    page_title="SPX 0DTE Journal",
    page_icon="📈",
    layout="wide",
)

# Initialize DB
@st.cache_resource
def init_db():
    create_tables()
    return True

init_db()

def main():
    st.sidebar.title("SPX 0DTE Journal")
    
    pages = {
        "Dashboard": ui.render_dashboard,
        "New Trade": ui.render_new_trade,
        "Trade Viewer": ui.render_trade_viewer,
        "AI Reports": ui.render_reports,
        "Settings & Keys": ui.render_settings
    }
    
    selection = st.sidebar.radio("Navigation", list(pages.keys()))
    
    # Render selected page
    pages[selection]()

if __name__ == "__main__":
    main()
