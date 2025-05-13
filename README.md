# 🚀 NLP Toolkit Application

A comprehensive toolkit for natural language processing tasks including data crawling, preprocessing, augmentation, model training, movie recommendations, and an AI chatbot.

**Author:** Toan Tran Cong 👨‍💻

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
  - [Windows Installation](#windows-installation)
  - [Linux Installation](#linux-installation)
- [Starting the Application](#starting-the-application)
- [Feature Guides](#feature-guides)
  - [🕸️ Data Crawler](#data-crawler)
  - [🔄 Data Pipeline](#data-pipeline)
  - [🔍 Data Augmentation](#data-augmentation)
  - [🧠 Model Training](#model-training)
  - [🎬 Movie Recommendation](#movie-recommendation)
  - [🤖 AI Chatbot](#ai-chatbot)
- [Troubleshooting](#troubleshooting)

## System Requirements

- **Python**: 3.9.18 (required version) 🐍
- **RAM**: 8GB minimum (16GB recommended for model training) 💾
- **Disk Space**: 2GB free space for installation and data 💽
- **Internet Connection**: Required for API-dependent features 🌐
- **Optional**: CUDA-capable GPU for faster model training 🖥️

## Installation Guide

### Windows Installation

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/toantc1024/NLP-Toolkit.git
   ```

   Or download and extract the ZIP file from https://github.com/toantc1024/NLP-Toolkit

2. **Set up a virtual environment (recommended)**

   ```bash
   # Navigate to the project directory
   cd path\to\NLP-Toolkit

   # Create a virtual environment with Python 3.9.18
   python -m venv venv

   # Activate the virtual environment
   venv\Scripts\activate
   ```

3. **Install required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   This may take a few minutes as some packages include large model files.

4. **Set up environment variables**
   Create a `.env` file in the root directory with the following variables:
   ```
   OMDB_API_KEY=your_omdb_api_key
   GOOGLE_GEN_AI_API_KEY=your_google_api_key
   ```
   - Get an OMDB API key from [OMDb API](https://www.omdbapi.com/apikey.aspx)
   - Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/)

### Linux Installation

1. **Clone or download the repository**

   ```bash
   git clone https://github.com/toantc1024/NLP-Toolkit.git
   ```

2. **Set up a virtual environment (recommended)**

   ```bash
   # Navigate to the project directory
   cd path/to/NLP-Toolkit

   # Ensure Python 3.9.18 is installed
   python3.9 --version

   # Create a virtual environment
   python3.9 -m venv venv

   # Activate the virtual environment
   source venv/bin/activate
   ```

3. **Install required dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   ```bash
   # Create .env file
   echo "OMDB_API_KEY=your_omdb_api_key" > .env
   echo "GOOGLE_GEN_AI_API_KEY=your_google_api_key" >> .env
   ```

5. **Additional Linux requirements for web crawling**

   ```bash
   # Install Chrome for Selenium
   sudo apt update
   sudo apt install -y chromium-browser

   # Install required system dependencies
   sudo apt install -y build-essential python3-dev
   ```

## Starting the Application

1. **Activate your virtual environment** (if not already activated)

   - Windows: `venv\Scripts\activate`
   - Linux: `source venv/bin/activate`

2. **Run the Streamlit application**

   ```bash
   streamlit run app.py
   ```

3. **Access the application**
   The application will automatically open in your default web browser, or you can access it at:
   ```
   http://localhost:8501
   ```

## Feature Guides

### 🕸️ Data Crawler

The Data Crawler allows you to extract data from websites like IMDB and Amazon using CSS selectors.

#### How to Use:

1. Select the website type from the dropdown (IMDB Reviews, Amazon Reviews, or Custom)
2. Enter the URL of the page to crawl
   - For IMDB reviews, use movie URLs like `https://www.imdb.com/title/tt0111161`
   - For Amazon reviews, use product pages like `https://www.amazon.com/product-name/dp/PRODUCTID/`
3. Adjust crawler settings:
   - For IMDB, the default selectors are pre-configured
   - For custom sites, you'll need to specify CSS selectors for items, titles, and content
4. Choose advanced options if needed:
   - Use direct browser control for sites that block automation
   - Add random delays to avoid detection
   - Set maximum items to crawl
5. Click "Start Web Crawling"
6. Once crawling is complete:
   - Review the extracted data
   - Rename columns if needed
   - Download the data as CSV
   - Or send it directly to the Data Pipeline tab

#### Tips:

- If you encounter "Access Denied" errors, try enabling "Use direct browser control"
- For complex sites, use developer tools (F12) to identify the correct CSS selectors
- Use the "Random delays" option to make crawling more human-like

### 🔄 Data Pipeline

The Data Pipeline allows you to import, preprocess, clean, and export data through a step-by-step process.

#### How to Use:

1. **Upload Data**:

   - Import data from CSV, Excel, or other formats
   - Or use data directly from the Data Crawler

2. **Select/Rename Columns**:

   - Choose which columns to keep in your dataset
   - Rename columns for clarity

3. **Preprocess Data**:

   - For text columns:
     - Select language (English or Vietnamese)
     - Apply text processing operations (remove punctuation, stopwords, etc.)
     - Tokenize text or convert to lowercase
   - For numeric columns:
     - Scale values (Min-Max)
     - Round to nearest decimal places
   - For date columns:
     - Extract year
     - Format dates

4. **Clean Data**:

   - Handle missing values (drop or replace)
   - Remove duplicate rows
   - View cleaning statistics

5. **Export Data**:
   - Download processed data in CSV, Excel, JSON, or Pickle format
   - Start a new pipeline if needed

#### Tips:

- Use the preview functionality at each step to verify transformations
- Vietnamese text processing works best when underthesea package is properly installed
- The step-by-step approach allows you to go back and adjust settings

### 🔍 Data Augmentation

Data Augmentation helps expand your dataset by creating variations of existing text entries.

#### How to Use:

1. **Upload Dataset**:

   - Upload a CSV or Excel file containing text data

2. **Select Columns**:

   - Choose the text column to augment
   - Optionally select a label column if your data is labeled

3. **Configure Label Descriptions** (for labeled data):

   - Provide detailed descriptions for each unique label
   - These descriptions help the system generate more accurate augmentations

4. **Choose Enhancement Settings**:

   - Select enhancement type:
     - Synonyms: Replace words with synonyms
     - Back Translation: Translate to another language and back
     - Word Embedding: Replace words with similar ones based on embeddings
     - Contextual Word Embeddings: Use BERT for context-aware replacements
     - Random Insertion/Swap/Deletion: Basic text manipulation
   - Set number of samples per text (1-5)
   - Enable/disable duplicate removal

5. **Preview Enhancement**:

   - Generate a preview on a small sample to verify quality
   - Compare original and enhanced texts

6. **Enhance Full Dataset**:

   - Process the entire dataset
   - View statistics about generated data

7. **Export Options**:
   - Export only enhanced data
   - Or merge with original data (with option to add source column)
   - Download as CSV or Excel

#### Tips:

- Contextual Word Embeddings generally produce the highest quality augmentations
- Back Translation works well for longer texts
- Always review the preview before processing the full dataset
- Start with fewer samples per text and increase if needed

### 🧠 Model Training

The Model Training feature allows you to train machine learning models for text classification.

#### How to Use:

1. **Train a New Model**:

   - Upload a labeled dataset (CSV or Excel)
   - Select content and label columns
   - Choose preprocessing steps (lowercase, remove punctuation, etc.)
   - Select vectorizers (up to 2):
     - TF-IDF: Traditional bag-of-words approach
     - BERT: Contextual embeddings
     - GloVe: Pre-trained word vectors
     - BGE M3: Multilingual embeddings
     - Word2Vec: Word embeddings
   - Select algorithms:
     - RandomForest, SVM, KNN, DecisionTree
     - CNN (requires embedding vectorizers, not TF-IDF)
   - Set train-test split ratio and random seed
   - Click "Train Models"
   - Review results:
     - Compare accuracy and F1 scores
     - Examine confusion matrices
     - Select best model for predictions
   - Test your model on new text inputs
   - Export your trained model for later use

2. **Load Model & Predict**:
   - Upload a previously trained model
   - Enter text to classify
   - Get predictions based on the loaded model

#### Tips:

- Different vectorizer-algorithm combinations work better for different tasks
- CNN requires embedding vectorizers (BERT, GloVe, BGE M3, Word2Vec)
- For small datasets, RandomForest often performs well
- For larger datasets with complex patterns, try CNN with BERT embeddings
- Always export your best model for future use

### 🎬 Movie Recommendation

The Movie Recommendation system offers personalized movie suggestions using collaborative filtering and content-based approaches.

#### How to Use:

1. **Collaborative Filtering**:

   - Select and rate up to 5 movies you've seen from the dropdown
   - Rate each movie from 1 (terrible) to 5 (excellent)
   - Click "Get Recommendations"
   - View personalized recommendations based on similar users' preferences
   - Click on movie posters or titles to view them on IMDB

2. **Content-Based Recommendation**:
   - Select a movie you like from the dropdown
   - Click "Find Similar Movies"
   - View recommendations based on genre similarity
   - See similarity scores and genre information

#### Tips:

- Rate more movies for better recommendations
- Be honest with your ratings for more accurate recommendations
- The collaborative filtering approach uses a memory-based user-item matrix
- Movies are recommended based on what similar users enjoyed

### 🤖 AI Chatbot

The AI Chatbot provides a conversational interface powered by Google's Gemini models, with ability to reference your own documents.

#### How to Use:

1. **Configuration**:

   - Choose API key source (system or custom)
   - Select Gemini model:
     - gemini-2.0-flash (fastest)
     - gemini-2.5-flash-preview (balanced)
     - gemini-2.5-pro-preview (most capable)
   - Enable/disable knowledge base
   - Adjust temperature (creativity level)

2. **Update Knowledge Base**:

   - Upload documents (PDF, TXT, DOCX)
   - Or enter a URL to extract content
   - Process documents to add them to the knowledge base
   - View status of knowledge base

3. **Chatbot Interface**:
   - Type questions in the chat input
   - View responses from the AI
   - Expand "View Source Documents" to see which parts of your documents were used
   - Chat history is maintained during your session

#### Tips:

- Enable "Use Knowledge Base" to get answers based on your documents
- Higher temperature values (closer to 1.0) make responses more creative but potentially less accurate
- Lower temperature values (closer to 0.0) make responses more deterministic and factual
- You need a Google Gemini API key for this feature to work
- The knowledge base is stored in memory and will reset when you reload the page

## Troubleshooting

### Common Issues:

1. **Installation Errors**:

   - Ensure you're using Python 3.9.18 exactly as specified
   - On Windows, you may need Microsoft Visual C++ Build Tools for some packages
   - On Linux, ensure you have the required system libraries installed

2. **CUDA/GPU Issues**:

   - If using GPU, ensure you have compatible CUDA drivers
   - Set appropriate environment variables for CUDA if needed

3. **Memory Errors**:

   - Reduce batch sizes when processing large datasets
   - Close other applications to free up system memory
   - For model training, try smaller vectorizers (TF-IDF instead of BERT)

4. **API Key Issues**:

   - Verify API keys are correctly set in the .env file
   - Check for API key usage limits (free tier restrictions)
   - Ensure proper format (no quotes or spaces) in the .env file

5. **Web Crawling Blocks**:

   - Use "direct browser control" option for sites with anti-scraping measures
   - Add random delays between requests
   - Consider using a VPN if IP is blocked

6. **Dependency Conflicts**:
   - Use a fresh virtual environment
   - Install dependencies in the order specified in requirements.txt

### Getting Help:

If you encounter issues not covered here, please:

1. Check the error logs (usually in the terminal where streamlit is running)
2. Ensure all dependencies are correctly installed
3. Create an issue on the GitHub repository: https://github.com/toantc1024/NLP-Toolkit/issues with:
   - Detailed description of the problem
   - Steps to reproduce
   - Complete error message
   - Your operating system and Python version

---

Made with ❤️ by Toan Tran Cong
