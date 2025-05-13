import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pickle
import os
import tempfile
from sentence_transformers import SentenceTransformer
import gensim.downloader as api
import torch
import warnings
# Add tensorflow and keras imports for CNN
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
# Import the BGE M3 embeddings from utils
from utils.Embedding import hf as bge_m3_embeddings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def app():
    st.title("Model Training")
    
    # Tabs for training or loading a model
    tab1, tab2 = st.tabs(["Train Model", "Load Model & Predict"])
    
    with tab1:
        train_model_tab()
    
    with tab2:
        load_model_tab()

def train_model_tab():
    st.header("Train a new model")
    
    # Initialize session state for models and related variables
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    if 'selected_model_key' not in st.session_state:
        st.session_state.selected_model_key = None
    if 'prediction_done' not in st.session_state:
        st.session_state.prediction_done = False
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'label_encoder' not in st.session_state:
        st.session_state.label_encoder = None
    if 'training_preprocess_options' not in st.session_state:
        st.session_state.training_preprocess_options = None
    
    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV or XLSX)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Load the data
        try:
            if uploaded_file.name.endswith('csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("File successfully loaded!")
            st.write("Preview of the dataset:")
            st.dataframe(df.head())
            
            # Column selection
            content_column = st.selectbox("Select the content column", df.columns.tolist())
            label_column = st.selectbox("Select the label column", df.columns.tolist())
            
            # Check for missing values
            missing_content = df[content_column].isnull().sum()
            missing_label = df[label_column].isnull().sum()
            
            if missing_content > 0 or missing_label > 0:
                st.warning(f"Warning: Missing values detected! Content column: {missing_content}, Label column: {missing_label}")
                handle_missing = st.radio("How would you like to handle missing values?", 
                                         ["Drop rows with missing values", "Cancel"])
                if handle_missing == "Drop rows with missing values":
                    df = df.dropna(subset=[content_column, label_column])
                    st.info(f"Dropped rows with missing values. New shape: {df.shape}")
                else:
                    st.stop()
            
            # Preprocessing options
            st.subheader("Preprocessing Options")
            
            preprocess_options = st.multiselect("Select preprocessing steps", 
                                              ["Lowercase", "Remove punctuation", "Remove numbers", 
                                               "Remove stopwords", "Stemming", "Lemmatization"])
            
            # Vectorizer selection
            st.subheader("Select Vectorizers (max 2)")
            vectorizers = st.multiselect("Choose vectorizers", 
                                        ["TF-IDF", "BERT", "GloVe", "BGE M3", "Word2Vec"], 
                                        default=["TF-IDF"])
            
            if len(vectorizers) > 2:
                st.error("Please select at most 2 vectorizers")
                st.stop()
            
            # Algorithm selection
            st.subheader("Select Algorithms")
            algorithms = st.multiselect("Choose algorithms", 
                                      ["RandomForest", "SVM", "KNN", "DecisionTree", "CNN"], 
                                      default=["RandomForest"])
            
            # Train-test split ratio
            test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
            
            # Random seed for reproducibility
            random_seed = st.number_input("Random seed", min_value=1, value=42)
            
            # Train the model
            if st.button("Train Models"):
                with st.spinner("Training models..."):
                    # Preprocess the text
                    processed_text = preprocess_text(df[content_column], preprocess_options)
                    
                    # Check for empty texts after preprocessing
                    empty_texts = [i for i, text in enumerate(processed_text) if not text.strip()]
                    if empty_texts:
                        st.warning(f"Warning: {len(empty_texts)} texts became empty after preprocessing. This might cause issues.")
                        # Remove empty texts
                        valid_indices = [i for i, text in enumerate(processed_text) if text.strip()]
                        processed_text = [processed_text[i] for i in valid_indices]
                        encoded_labels = encoded_labels[valid_indices] if len(valid_indices) < len(encoded_labels) else encoded_labels
                        st.info(f"Removed empty texts. New dataset size: {len(processed_text)}")
                    
                    # Ensure we have data left
                    if len(processed_text) == 0:
                        st.error("All texts are empty after preprocessing. Please try different preprocessing options.")
                        st.stop()
                    
                    # Encode labels
                    le = LabelEncoder()
                    encoded_labels = le.fit_transform(df[label_column])
                    class_names = le.classes_
                    
                    # Store the label encoder and preprocessing options in session state
                    st.session_state.label_encoder = le
                    st.session_state.training_preprocess_options = preprocess_options
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        processed_text, encoded_labels, test_size=test_size, random_state=random_seed
                    )
                    
                    # Results storage
                    results = {}
                    best_model = None
                    best_accuracy = 0
                    best_vectorizer = None
                    best_vectorizer_name = None
                    best_algorithm = None
                    confusion_matrices = {}
                    skipped_combinations = []
                    
                    # Train models for each vectorizer and algorithm combination
                    for vectorizer_name in vectorizers:
                        vectorizer = get_vectorizer(vectorizer_name)
                        
                        # For embedding-based vectorizers
                        try:
                            if vectorizer_name in ["BERT", "GloVe", "BGE M3", "Word2Vec"]:
                                X_train_vec = vectorize_text(X_train, vectorizer, vectorizer_name)
                                X_test_vec = vectorize_text(X_test, vectorizer, vectorizer_name)
                            else:  # For TF-IDF
                                vectorizer.fit(X_train)
                                X_train_vec = vectorizer.transform(X_train)
                                X_test_vec = vectorizer.transform(X_test)
                            
                            for algorithm_name in algorithms:
                                # Skip incompatible vectorizer-CNN combinations
                                if algorithm_name == "CNN" and vectorizer_name == "TF-IDF":
                                    combination = f"{vectorizer_name} + {algorithm_name}"
                                    skipped_combinations.append((combination, "CNN requires embedding vectorizers (BERT, GloVe, BGE M3, Word2Vec)"))
                                    st.warning(f"Skipping incompatible combination: {combination} - CNN requires embedding vectorizers")
                                    continue
                                
                                try:
                                    # Special handling for CNN
                                    if algorithm_name == "CNN":
                                        # Convert labels to categorical for CNN
                                        num_classes = len(np.unique(y_train))
                                        y_train_cat = to_categorical(y_train, num_classes)
                                        y_test_cat = to_categorical(y_test, num_classes)
                                        
                                        # Create and train CNN model
                                        model = get_model(algorithm_name, input_shape=X_train_vec.shape[1], num_classes=num_classes)
                                        
                                        # Train with early stopping to prevent overfitting
                                        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
                                        model.fit(
                                            X_train_vec, y_train_cat, 
                                            epochs=10, batch_size=32, 
                                            validation_split=0.1,
                                            callbacks=[early_stopping],
                                            verbose=0
                                        )
                                        
                                        # Make predictions
                                        y_pred_proba = model.predict(X_test_vec)
                                        y_pred = np.argmax(y_pred_proba, axis=1)
                                    else:
                                        model = get_model(algorithm_name)
                                        # Train the model
                                        model.fit(X_train_vec, y_train)
                                        # Make predictions
                                        y_pred = model.predict(X_test_vec)
                                    
                                    # Calculate metrics
                                    accuracy = accuracy_score(y_test, y_pred)
                                    f1 = f1_score(y_test, y_pred, average='weighted')
                                    conf_matrix = confusion_matrix(y_test, y_pred)
                                    
                                    # Store results
                                    model_name = f"{vectorizer_name} + {algorithm_name}"
                                    results[model_name] = {
                                        'accuracy': accuracy,
                                        'f1_score': f1,
                                        'model': model,
                                        'vectorizer': vectorizer,
                                        'vectorizer_name': vectorizer_name,
                                        'algorithm_name': algorithm_name
                                    }
                                    confusion_matrices[model_name] = conf_matrix
                                    
                                    # Check if this is the best model so far
                                    if accuracy > best_accuracy:
                                        best_accuracy = accuracy
                                        best_model = model
                                        best_vectorizer = vectorizer
                                        best_vectorizer_name = vectorizer_name
                                        best_algorithm = algorithm_name
                                except Exception as model_error:
                                    # Handle model-specific errors
                                    combination = f"{vectorizer_name} + {algorithm_name}"
                                    skipped_combinations.append((combination, str(model_error)))
                                    st.warning(f"Skipping incompatible combination: {combination}")
                                    continue
                        except Exception as vec_error:
                            # Handle vectorizer-specific errors
                            for algorithm_name in algorithms:
                                combination = f"{vectorizer_name} + {algorithm_name}"
                                skipped_combinations.append((combination, str(vec_error)))
                            st.warning(f"Skipping vectorizer {vectorizer_name} due to error: {str(vec_error)}")
                            continue
                    
                    # Check if we have any results
                    if not results:
                        st.error("All model-vectorizer combinations failed. Please try different options.")
                        # Display skipped combinations
                        if skipped_combinations:
                            st.subheader("Skipped Combinations")
                            for combination, error in skipped_combinations:
                                st.error(f"{combination}: {error}")
                        st.stop()
                    
                    # Store all trained models in session state for later selection
                    st.session_state.trained_models = results
                    
                    # Display information about skipped combinations
                    if skipped_combinations:
                        st.subheader("Skipped Combinations")
                        st.info(f"{len(skipped_combinations)} model-vectorizer combinations were skipped due to compatibility issues.")
                        with st.expander("View details"):
                            for combination, error in skipped_combinations:
                                st.error(f"{combination}: {error}")
                    
                    # Display results
                    st.subheader("Model Comparison")
                    
                    # Create a DataFrame for results
                    results_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'Accuracy': [results[model]['accuracy'] for model in results],
                        'F1 Score': [results[model]['f1_score'] for model in results]
                    })
                    
                    # Sort by accuracy
                    results_df = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
                    
                    # Display results
                    st.dataframe(results_df)
                    
                    # Plotting accuracy and F1 score
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bar_width = 0.35
                    index = np.arange(len(results_df))
                    
                    bar1 = ax.bar(index, results_df['Accuracy'], bar_width, label='Accuracy')
                    bar2 = ax.bar(index + bar_width, results_df['F1 Score'], bar_width, label='F1 Score')
                    
                    ax.set_xlabel('Models')
                    ax.set_ylabel('Scores')
                    ax.set_title('Model Comparison')
                    ax.set_xticks(index + bar_width / 2)
                    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Display confusion matrices
                    st.subheader("Confusion Matrices")
                    
                    for model_name, conf_matrix in confusion_matrices.items():
                        st.write(f"**{model_name}**")
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                                   xticklabels=class_names, yticklabels=class_names)
                        plt.title(f'Confusion Matrix - {model_name}')
                        plt.ylabel('True Label')
                        plt.xlabel('Predicted Label')
                        st.pyplot(fig)
                    
                    # Highlight the best model
                    st.subheader("Best Model")
                    st.info(f"The best model is {best_vectorizer_name} + {best_algorithm} with an accuracy of {best_accuracy:.4f}")
                    
                    # Set the best model as the default selected model
                    best_model_key = f"{best_vectorizer_name} + {best_algorithm}"
                    st.session_state.selected_model_key = best_model_key
            
            # Prediction section - only show after models are trained
            if st.session_state.trained_models and st.session_state.label_encoder is not None:
                st.subheader("Test Your Models")
                
                # Let user select which model to use for prediction
                model_options = list(st.session_state.trained_models.keys())
                selected_model_key = st.selectbox(
                    "Select model for prediction", 
                    model_options,
                    index=model_options.index(st.session_state.selected_model_key) if st.session_state.selected_model_key in model_options else 0
                )
                
                # Update the selected model key in session state
                st.session_state.selected_model_key = selected_model_key
                
                # Get the selected model details
                selected_model = st.session_state.trained_models[selected_model_key]['model']
                selected_vectorizer = st.session_state.trained_models[selected_model_key]['vectorizer']
                selected_vectorizer_name = st.session_state.trained_models[selected_model_key]['vectorizer_name']
                
                # Display model accuracy
                st.info(f"Selected model accuracy: {st.session_state.trained_models[selected_model_key]['accuracy']:.4f}")
                
                # Text input for prediction
                test_text = st.text_area("Enter text to classify", "")
                
                # Prediction button
                if st.button("Predict", key="predict_button"):
                    if test_text:
                        with st.spinner("Predicting..."):
                            # Preprocess the input text using stored preprocessing options
                            processed_input = preprocess_text([test_text], st.session_state.training_preprocess_options)
                            
                            # Vectorize
                            if selected_vectorizer_name in ["BERT", "GloVe", "BGE M3", "Word2Vec"]:
                                input_vec = vectorize_text(processed_input, selected_vectorizer, selected_vectorizer_name)
                            else:  # For TF-IDF
                                input_vec = selected_vectorizer.transform(processed_input)
                            
                            # Predict
                            prediction = selected_model.predict(input_vec)
                            # Use the label encoder from session state
                            predicted_class = st.session_state.label_encoder.inverse_transform(prediction)[0]
                            
                            # Store the prediction result in session state
                            st.session_state.prediction_done = True
                            st.session_state.prediction_result = predicted_class
                
                # Display prediction result if available
                if st.session_state.prediction_done and st.session_state.prediction_result:
                    st.success(f"Predicted class: {st.session_state.prediction_result}")
                
                # Export model section
                st.subheader("Export Model")
                export_model_option = st.radio(
                    "Which model do you want to export?",
                    ["Selected model", "Best model"],
                    index=0
                )
                
                model_name = st.text_input("Model filename", "my_model")
                
                if st.button("Export Model"):
                    # Determine which model to export
                    if export_model_option == "Selected model":
                        export_model = selected_model
                        export_vectorizer = selected_vectorizer
                        export_vectorizer_name = selected_vectorizer_name
                        export_algorithm = st.session_state.trained_models[selected_model_key]['algorithm_name']
                    else:  # Best model
                        export_model = best_model
                        export_vectorizer = best_vectorizer
                        export_vectorizer_name = best_vectorizer_name
                        export_algorithm = best_algorithm
                    
                    # Create a model package with all necessary components
                    model_package = {
                        'model': export_model,
                        'vectorizer': export_vectorizer,
                        'vectorizer_name': export_vectorizer_name,
                        'algorithm_name': export_algorithm,
                        'label_encoder': st.session_state.label_encoder,
                        'preprocess_options': st.session_state.training_preprocess_options
                    }
                    
                    if not model_name.endswith('.pkl'):
                        model_name += '.pkl'
                    
                    # Save the model package
                    with open(model_name, 'wb') as f:
                        pickle.dump(model_package, f)
                    
                    # Provide the download button
                    with open(model_name, 'rb') as f:
                        model_bytes = f.read()
                        
                    st.download_button(
                        label="Download Model",
                        data=model_bytes,
                        file_name=model_name,
                        mime="application/octet-stream"
                    )
                    
                    st.success(f"Model exported as {model_name}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.exception(e)

def load_model_tab():
    st.header("Load Model & Predict")
    
    # Reset prediction state when switching to this tab
    if 'load_prediction_done' not in st.session_state:
        st.session_state.load_prediction_done = False
    if 'load_prediction_result' not in st.session_state:
        st.session_state.load_prediction_result = None
    
    # File upload for model
    uploaded_model = st.file_uploader("Upload your trained model (.pkl)", type=['pkl'])
    
    if uploaded_model is not None:
        try:
            # Save the uploaded model to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
                tmp_file.write(uploaded_model.getvalue())
                model_path = tmp_file.name
            
            # Load the model package
            with open(model_path, 'rb') as f:
                model_package = pickle.load(f)
            
            # Extract components
            model = model_package['model']
            vectorizer = model_package['vectorizer']
            vectorizer_name = model_package['vectorizer_name']
            algorithm_name = model_package['algorithm_name']
            label_encoder = model_package['label_encoder']
            preprocess_options = model_package['preprocess_options']
            
            st.success(f"Model loaded successfully: {vectorizer_name} + {algorithm_name}")
            
            # Text input for prediction
            test_text = st.text_area("Enter text to classify", "")
            
            if st.button("Predict", key="load_predict_button"):
                if test_text:
                    with st.spinner("Predicting..."):
                        # Preprocess the input text
                        processed_input = preprocess_text([test_text], preprocess_options)
                        
                        # Vectorize
                        if vectorizer_name in ["BERT", "GloVe", "BGE M3", "Word2Vec"]:
                            input_vec = vectorize_text(processed_input, vectorizer, vectorizer_name)
                        else:  # For TF-IDF
                            input_vec = vectorizer.transform(processed_input)
                        
                        # Predict
                        prediction = model.predict(input_vec)
                        predicted_class = label_encoder.inverse_transform(prediction)[0]
                        
                        # Store the prediction result in session state
                        st.session_state.load_prediction_done = True
                        st.session_state.load_prediction_result = predicted_class
            
            # Display prediction result if available
            if st.session_state.load_prediction_done and st.session_state.load_prediction_result:
                st.success(f"Predicted class: {st.session_state.load_prediction_result}")
            
            # Clean up temporary file
            os.unlink(model_path)
            
        except Exception as e:
            st.error(f"Error loading the model: {str(e)}")
            st.exception(e)

def preprocess_text(texts, options):
    """Preprocess text based on selected options"""
    processed_texts = texts.copy()
    
    for i, text in enumerate(processed_texts):
        # Ensure text is string
        if not isinstance(text, str):
            text = str(text)
        
        # Apply selected preprocessing steps
        if "Lowercase" in options:
            text = text.lower()
        
        if "Remove punctuation" in options:
            text = re.sub(r'[^\w\s]', '', text)
        
        if "Remove numbers" in options:
            text = re.sub(r'\d+', '', text)
        
        # Tokenize for word-level operations
        tokens = nltk.word_tokenize(text)
        
        if "Remove stopwords" in options:
            stop_words = set(stopwords.words('english'))
            tokens = [word for word in tokens if word not in stop_words]
        
        if "Stemming" in options:
            stemmer = PorterStemmer()
            tokens = [stemmer.stem(word) for word in tokens]
        
        if "Lemmatization" in options:
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        # Join tokens back to text
        processed_texts[i] = ' '.join(tokens)
    
    return processed_texts

def get_vectorizer(vectorizer_name):
    """Get the vectorizer based on the name"""
    if vectorizer_name == "TF-IDF":
        return TfidfVectorizer(max_features=5000)
    elif vectorizer_name == "BERT":
        try:
            # Try loading a smaller BERT model that's more memory efficient
            return SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller, more efficient model
        except Exception as e:
            # Log the actual error for debugging
            st.error(f"Error loading BERT model: {str(e)}")
            # Try an even smaller fallback model
            try:
                return SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
            except:
                # If all else fails, use a very small model
                return SentenceTransformer('all-MiniLM-L6-v2')
    elif vectorizer_name == "GloVe":
        return api.load("glove-wiki-gigaword-100")
    elif vectorizer_name == "BGE M3":
        # Use the pre-loaded BGE M3 model from utils/Embedding.py
        return bge_m3_embeddings
    elif vectorizer_name == "Word2Vec":
        return api.load("word2vec-google-news-300")
    else:
        raise ValueError(f"Unsupported vectorizer: {vectorizer_name}")

def vectorize_text(texts, vectorizer, vectorizer_name):
    """Vectorize text based on the vectorizer type"""
    try:
        if vectorizer_name == "BERT":
            # Use batch processing for transformer models to reduce memory usage
            try:
                # Default batch size
                batch_size = 32
                
                # Check if we need to batch (for larger datasets)
                if len(texts) > batch_size:
                    # Process in batches to avoid memory issues
                    all_embeddings = []
                    for i in range(0, len(texts), batch_size):
                        batch = texts[i:i+batch_size]
                        batch_embeddings = vectorizer.encode(batch, convert_to_numpy=True)
                        all_embeddings.append(batch_embeddings)
                    # Combine all batches
                    return np.vstack(all_embeddings)
                else:
                    # For smaller datasets, process all at once
                    return vectorizer.encode(texts, convert_to_numpy=True)
            
            except Exception as bert_error:
                # Provide detailed error information
                error_message = f"Error processing {vectorizer_name} embeddings: {str(bert_error)}"
                
                # Try with a smaller batch size as a fallback
                try:
                    smaller_batch_size = 8
                    st.warning(f"Attempting with smaller batch size ({smaller_batch_size})...")
                    
                    all_embeddings = []
                    for i in range(0, len(texts), smaller_batch_size):
                        batch = texts[i:i+smaller_batch_size]
                        batch_embeddings = vectorizer.encode(batch, convert_to_numpy=True)
                        all_embeddings.append(batch_embeddings)
                    return np.vstack(all_embeddings)
                
                except Exception as fallback_error:
                    # If both attempts fail, raise a comprehensive error
                    raise ValueError(f"{error_message}\nFallback also failed: {str(fallback_error)}")
        
        elif vectorizer_name == "BGE M3":
            # Use the HuggingFaceEmbeddings implementation for BGE M3
            try:
                # Process in smaller batches to avoid memory issues
                batch_size = 16
                all_embeddings = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    # Use embed_query for single texts or embed_documents for batch processing
                    if len(batch) == 1:
                        batch_embeddings = [vectorizer.embed_query(batch[0])]
                    else:
                        batch_embeddings = vectorizer.embed_documents(batch)
                    
                    all_embeddings.extend(batch_embeddings)
                
                # Convert to numpy array
                embeddings_array = np.array(all_embeddings)
                return embeddings_array
            
            except Exception as bge_error:
                # If original approach fails, try an even simpler approach
                try:
                    st.warning("Attempting simpler embedding approach for BGE M3...")
                    embeddings = []
                    for text in texts:
                        # Process one text at a time
                        embedding = vectorizer.embed_query(text)
                        embeddings.append(embedding)
                    
                    return np.array(embeddings)
                
                except Exception as fallback_error:
                    raise ValueError(f"Error processing BGE M3 embeddings: {str(bge_error)}\nFallback also failed: {str(fallback_error)}")
        
        elif vectorizer_name == "GloVe" or vectorizer_name == "Word2Vec":
            # Process all texts
            vector_size = vectorizer.vector_size
            embeddings = []
            
            for text in texts:
                # Tokenize
                words = text.split()
                
                # Get word vectors and average them
                word_vecs = []
                for word in words:
                    if word in vectorizer:
                        word_vecs.append(vectorizer[word])
                
                if word_vecs:
                    # If we have word vectors, average them
                    avg_vec = np.mean(word_vecs, axis=0)
                else:
                    # If no words found in the embedding, use zeros
                    avg_vec = np.zeros(vector_size)
                
                embeddings.append(avg_vec)
            
            # Ensure all embeddings have the same shape and return as numpy array
            embeddings_array = np.array(embeddings)
            
            # Check for NaN or infinity values and replace with zeros
            if not np.all(np.isfinite(embeddings_array)):
                embeddings_array = np.nan_to_num(embeddings_array)
                
            return embeddings_array
        
        else:
            raise ValueError(f"Unsupported vectorizer for vectorize_text: {vectorizer_name}")
    
    except Exception as e:
        # Add more detail to the error message to better diagnose issues
        error_detail = str(e)
        if "memory" in error_detail.lower():
            suggestion = " - Try reducing batch size or using a smaller model"
        elif "cuda" in error_detail.lower():
            suggestion = " - CUDA/GPU issue detected"
        else:
            suggestion = " - Check your input data and model compatibility"
        
        raise ValueError(f"Error in vectorize_text with {vectorizer_name}: {error_detail}{suggestion}")

def get_model(algorithm_name, input_shape=None, num_classes=None):
    """Get the model based on the algorithm name"""
    if algorithm_name == "RandomForest":
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif algorithm_name == "SVM":
        return SVC(probability=True, random_state=42)
    elif algorithm_name == "KNN":
        return KNeighborsClassifier(n_neighbors=5)
    elif algorithm_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=42)
    elif algorithm_name == "CNN":
        if input_shape is None or num_classes is None:
            raise ValueError("CNN requires input_shape and num_classes parameters")
        
        # Create a simple CNN model
        model = Sequential([
            # Reshape input for Conv1D (add channel dimension)
            tf.keras.layers.Reshape((input_shape, 1), input_shape=(input_shape,)),
            Conv1D(filters=64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(100, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")

if __name__ == "__main__":
    app()