from st_on_hover_tabs import on_hover_tabs
from utils import DataEnhance
import streamlit as st
from tabs import DataPipeline, DataCrawler, DataAugmentation, ModelTraining, MovieRecommendation, Chatbot  # Added Chatbot import

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
        tab_names = ['Dashboard', 'Data Pipeline', 'Data crawler', 'Data Augmentation', 'Model Training', 'Movie Recommendation', 'Chatbot']
        if current_tab in tab_names:
            default_idx = tab_names.index(current_tab)
        else:
            default_idx = 0
        # Reset the active tab
        st.session_state.active_tab = None
    else:
        default_idx = 0
        
    # Display tabs with the determined default index
    tabs = on_hover_tabs(tabName=['Dashboard', 'Data crawler', 'Data Pipeline', 'Data Augmentation', 'Model Training', 'Movie Recommendation', 'Chatbot'], 
                         iconName=['dashboard', 'analytics', 'bolt', 'rocket', 'science', 'movie_filter', 'chat'], 
                         default_choice=default_idx)

# Display the selected tab content
if tabs == 'Dashboard':
    st.title("Dashboard")
    st.write('Welcome to the NLP project dashboard')
    
    st.markdown("""
    ## 📚 NLP Toolbox Overview
    
    This application provides a complete data pipeline for NLP projects, from data collection to model training and visualization.
    
    Select a tab from the navigation bar on the left to get started with your NLP journey.
    """)
    
    # Data Pipeline tutorial
    with st.expander("🔄 Data Pipeline Tutorial", expanded=False):
        st.markdown("""
        ### Data Pipeline
        
        The Data Pipeline tab helps you prepare and clean your text data for NLP tasks.
        
        #### Step-by-step guide:
        
        1. **Upload Data**:
           - Upload CSV, Excel, TSV, or TXT files
           - Specify delimiters for CSV/TSV files
           - Select specific sheets for Excel files
        
        2. **Select/Rename Columns**:
           - Choose which columns to keep in your dataset
           - Rename columns to more appropriate names
        
        3. **Preprocess Data**:
           - Apply text preprocessing techniques:
               - Remove punctuation, numbers, or stopwords
               - Convert text to lowercase
               - Remove HTML tags or URLs
               - Tokenize text
           - Set language-specific processing (English or Vietnamese)
           - Handle numeric and datetime columns appropriately
        
        4. **Clean Data**:
           - Handle missing values with multiple strategies
           - Remove duplicate entries
           - Filter data based on criteria
        
        5. **Export Data**:
           - Export to CSV, Excel, JSON, or Pickle formats
           - Download processed data for use in other applications
        
        **Tip**: You can also receive data directly from the Data Crawler tab by clicking "Send to Data Pipeline".
        """)
    
    # Data Crawler tutorial
    with st.expander("🌐 Data Crawler Tutorial", expanded=False):
        st.markdown("""
        ### Data Crawler
        
        The Data Crawler tab allows you to collect data from websites without writing code.
        
        #### Step-by-step guide:
        
        1. **Select Website Type**:
           - Choose from pre-configured options (IMDB Reviews, Amazon Reviews)
           - Or select "Generic Site" for custom crawling
        
        2. **Enter URL**:
           - Provide the webpage URL you want to crawl
           - For IMDB, the movie page will be automatically navigated to its reviews page
        
        3. **Configure Selectors**:
           - Each web element is targeted using CSS selectors
           - Pre-configured for popular sites, or customize for your target site:
               - Item selector: the container for each item/review
               - Title selector: where to find the title text
               - Content selector: where to find the main content
               - Rating selector: where to find rating information
        
        4. **Advanced Options**:
           - Adjust crawler behavior (headless mode, scrolling, timeouts)
           - Set anti-detection options to avoid being blocked
           - Configure PDF handling options when relevant
        
        5. **Export or Transfer Data**:
           - Download the crawled data as CSV
           - Rename columns before exporting
           - Send directly to the Data Pipeline for further processing
        
        **Tip**: For complex websites, try enabling "Use direct browser control" in the Advanced Options.
        """)
    
    # Data Augmentation tutorial
    with st.expander("🚀 Data Augmentation Tutorial", expanded=False):
        st.markdown("""
        ### Data Augmentation
        
        The Data Augmentation tab helps you expand your dataset by creating variations of your existing text data.
        
        #### Step-by-step guide:
        
        1. **Upload Data**:
           - Upload your CSV or Excel file containing text data
        
        2. **Select Columns**:
           - Choose the text column to enhance
           - Optionally select a label column if working with classified data
           - Provide descriptions for labels to improve predictions
        
        3. **Choose Enhancement Settings**:
           - Select enhancement type:
               - Synonym replacement
               - Word insertion
               - Word deletion
               - Back translation
               - etc.
           - Set number of samples to generate per text
           - Toggle duplicate removal option
        
        4. **Preview Enhancement**:
           - Generate a preview on a small sample of your data
           - Compare original and enhanced texts
        
        5. **Enhance Full Dataset**:
           - Apply enhancement to the entire dataset
           - Export options:
               - Enhanced data only
               - Combined with original data
               - Add source identifier column
        
        **Tip**: For classification tasks, providing detailed label descriptions dramatically improves the quality of augmented data.
        """)
    
    # Model Training tutorial
    with st.expander("🧠 Model Training Tutorial", expanded=False):
        st.markdown("""
        ### Model Training
        
        The Model Training tab lets you train and evaluate NLP models without writing code.
        
        #### Step-by-step guide:
        
        1. **Upload Training Data**:
           - Upload a CSV or Excel file with text and labels
        
        2. **Select Columns**:
           - Choose which column contains the text
           - Select the label/target column
        
        3. **Preprocessing Options**:
           - Select text cleaning operations
           - Choose between various preprocessing techniques
        
        4. **Select Vectorizers and Algorithms**:
           - Choose vectorization methods:
               - TF-IDF
               - BERT, GloVe, Word2Vec, BGE M3
           - Select machine learning algorithms:
               - RandomForest, SVM, KNN, DecisionTree, CNN
        
        5. **Train and Evaluate Models**:
           - View performance metrics (accuracy, F1 score)
           - Compare different models
           - Visualize confusion matrices
        
        6. **Test and Export Model**:
           - Try your model on new text inputs
           - Export the trained model for later use
           - Load saved models to make predictions
        
        **Tip**: Try different combinations of vectorizers and algorithms to find the best performance for your specific task.
        """)
    
    # Movie Recommendation tutorial
    with st.expander("🎬 Movie Recommendation Tutorial", expanded=False):
        st.markdown("""
        ### Movie Recommendation
        
        The Movie Recommendation tab demonstrates recommendation systems using collaborative and content-based filtering.
        
        #### Step-by-step guide:
        
        1. **Collaborative Filtering**:
           - Select and rate up to 5 movies you've watched
           - Get recommendations based on similar users' preferences
           - View movie posters, ratings, and details
           - Click on movie titles or posters to view on IMDB
        
        2. **Content-Based Recommendation**:
           - Select a single movie you like
           - Get recommendations based on movie characteristics
           - See similarity scores and genre matching
        
        **How it works**: 
        - **Collaborative filtering** finds users with similar tastes to yours and recommends what they liked
        - **Content-based filtering** recommends movies with similar genres, actors, directors, etc.
        
        **Tip**: Rate a diverse set of movies to get more varied recommendations.
        """)
    
    # Chatbot tutorial
    with st.expander("💬 Chatbot Tutorial", expanded=False):
        st.markdown("""
        ### AI Chatbot with Knowledge Base
        
        The Chatbot tab provides an AI assistant that can answer questions based on your documents.
        
        #### Step-by-step guide:
        
        1. **Configuration**:
           - Choose API key source (system or custom)
           - Select the Gemini model to use
           - Adjust temperature for more creative or conservative responses
           - Toggle knowledge base usage
        
        2. **Update Knowledge Base**:
           - Upload documents (PDF, TXT, DOCX)
           - Or enter a URL to extract content
           - Process documents to create a searchable knowledge base
        
        3. **Chat Interface**:
           - Ask questions about your documents
           - View sources of information used for answers
           - Track token usage
           - Clear chat history as needed
        
        **How it works**: The system splits documents into chunks, creates embeddings, and uses semantic search to find relevant information when answering your questions.
        
        **Tip**: For the best results, upload focused documents related to your specific domain or topic.
        """)
    
elif tabs == 'Data Pipeline':
    # Load the data pipeline tab
    DataPipeline.app()

elif tabs == 'Data crawler':
    # Load the data crawler tab
    DataCrawler.app()

elif tabs == 'Data Augmentation':
    # Load the data augmentation tab
    DataAugmentation.app()

elif tabs == 'Model Training':
    # Load the model training tab
    ModelTraining.app()

elif tabs == 'Movie Recommendation':
    # Load the movie recommendation tab
    MovieRecommendation.app()

elif tabs == 'Chatbot':
    # Load the chatbot tab
    Chatbot.app()

elif tabs == 'Analysis':
    st.title("Analysis")
    st.write('Text analysis features will be available here')
    st.info("This section will include NLP analysis features such as tokenization, stemming, lemmatization, and more.")

elif tabs == 'Visualization':
    st.title("Visualization")
    st.write('Data visualization features will be available here')
    st.info("This section will include visualizations like word clouds, frequency charts, sentiment analysis results, etc.")
