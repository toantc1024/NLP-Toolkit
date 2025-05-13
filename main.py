from st_on_hover_tabs import on_hover_tabs
import streamlit as st
from tabs import DataPipeline, DataCrawler  # Import our modules

# Set page configuration
st.set_page_config(layout="wide")

# Load stylesheet
st.markdown('<style>' + open('./style.css').read() + '</style>', unsafe_allow_html=True)

# Application title
st.header("NLP Toolbox")

# Initialize active tab in session state if not exists
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = None

# Sidebar with hover tabs
with st.sidebar:
    # Get the current tab selection or use the active_tab from session_state if set
    current_tab = st.session_state.active_tab
    
    # If active_tab is set in session state, use it and then reset it
    if current_tab:
        # Find the index of the tab
        tab_names = ['Dashboard', 'Data Pipeline', 'Data crawler']
        if current_tab in tab_names:
            default_idx = tab_names.index(current_tab)
        else:
            default_idx = 0
        # Reset the active tab
        st.session_state.active_tab = None
    else:
        default_idx = 0
        
    # Display tabs with the determined default index
    tabs = on_hover_tabs(tabName=['Dashboard', 'Data crawler', 'Data Pipeline', ], 
                         iconName=['dashboard', 'analytics', 'bolt'], 
                         default_choice=default_idx)

# Display the selected tab content
if tabs == 'Dashboard':
    st.title("Dashboard")
    st.write('Welcome to the NLP project dashboard')
    st.info("""
    This application provides a complete data pipeline for NLP projects:
    
    1. **Data Pipeline**: Upload, preprocess, clean, and export your data
    2. **Analysis**: Apply NLP techniques to your data
    3. **Visualization**: Visualize your results
    4. **Data Crawler**: Collect data from websites like IMDB or extract from PDFs
    
    Select a tab from the navigation bar to get started.
    """)

elif tabs == 'Data Pipeline':
    # Load the data pipeline tab
    DataPipeline.app()

elif tabs == 'Data crawler':
    # Load the data crawler tab
    DataCrawler.app()

elif tabs == 'Analysis':
    st.title("Analysis")
    st.write('Text analysis features will be available here')
    st.info("This section will include NLP analysis features such as tokenization, stemming, lemmatization, and more.")

elif tabs == 'Visualization':
    st.title("Visualization")
    st.write('Data visualization features will be available here')
    st.info("This section will include visualizations like word clouds, frequency charts, sentiment analysis results, etc.")
