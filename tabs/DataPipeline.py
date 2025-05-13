import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import re
import string
import emoji
import time
from io import StringIO

# NLTK imports
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    st.warning("NLTK download failed. Some NLP features may not work properly.")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Underthesea for Vietnamese language processing
try:
    from underthesea import word_tokenize as vi_tokenize
    from underthesea import pos_tag, sent_tokenize
    underthesea_available = True
except ImportError:
    underthesea_available = False
    st.warning("Underthesea package not available. Vietnamese text processing will be limited.")

def app():
    st.title("Data Pipeline")
    
    # Create a multi-step form
    pipeline_steps = ["Upload Data", "Select/Rename Columns", "Preprocess Data", "Clean Data", "Export Data"]
    
    # Initialize session state for storing data between steps
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'step' not in st.session_state:
        st.session_state.step = 0
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'selected_cols' not in st.session_state:
        st.session_state.selected_cols = None
    if 'renamed_cols' not in st.session_state:
        st.session_state.renamed_cols = {}
    if 'pre_processed' not in st.session_state:
        st.session_state.pre_processed = False
    if 'cleaned' not in st.session_state:
        st.session_state.cleaned = False
        
    # Create the progress bar
    progress_bar = st.progress(0)
    
    # Display the current step
    st.subheader(f"Step {st.session_state.step + 1}: {pipeline_steps[st.session_state.step]}")
    
    # Step 1: Upload Data
    if st.session_state.step == 0:
        upload_data()
        
    # Step 2: Select/Rename Columns
    elif st.session_state.step == 1:
        select_rename_columns()
        
    # Step 3: Preprocess Data
    elif st.session_state.step == 2:
        preprocess_data()
        
    # Step 4: Clean Data
    elif st.session_state.step == 3:
        clean_data()
        
    # Step 5: Export Data
    elif st.session_state.step == 4:
        export_data()
    
    # Display navigation buttons
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.step > 0:
            if st.button("Previous Step"):
                st.session_state.step -= 1
                progress_bar.progress((st.session_state.step) / (len(pipeline_steps) - 1))
                st.rerun()
    
    with col2:
        if st.session_state.step < len(pipeline_steps) - 1:
            # Enable the Next button based on conditions
            next_disabled = False
            if st.session_state.step == 0 and st.session_state.data is None:
                next_disabled = True
            
            if not next_disabled and st.button("Next Step"):
                st.session_state.step += 1
                progress_bar.progress((st.session_state.step) / (len(pipeline_steps) - 1))
                st.rerun()

def upload_data():
    """Step 1: Upload different types of data files"""
    
    # Check if data is available from the DataCrawler
    if 'pipeline_df' in st.session_state and st.session_state.pipeline_df is not None:
        st.info("Data detected from the Data Crawler")
        
        # Show preview of the imported data
        st.subheader("Data from Crawler")
        df = st.session_state.pipeline_df
        st.dataframe(df.head())
        
        # Display data info
        st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        
        # Display detailed stats about the data
        with st.expander("View detailed data statistics"):
            buffer = io.StringIO()
            df.info(buf=buffer)
            info_str = buffer.getvalue()
            st.text(info_str)
            
            # Show data types and missing values
            st.subheader("Data Types and Missing Values")
            data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
            missing_values = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
            data_info = pd.concat([data_types, missing_values], axis=1)
            st.dataframe(data_info)
        
        # Ask user if they want to use this data
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Use data from Data Crawler", key="use_crawler_data"):
                # Store the data in session state
                st.session_state.data = df
                st.session_state.processed_data = df.copy()
                
                # Show origin info in processing log
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                # Add log entry with timestamp
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.processing_log.append(f"[{timestamp}] Imported {df.shape[0]} rows from Data Crawler")
                
                # Show success message
                st.success("Successfully imported data from crawler!")
                # Force rerun to update the UI
                st.rerun()
        
        with col2:
            if st.button("Clear crawler data", key="clear_crawler_data"):
                # Clear the crawler data
                st.session_state.pipeline_df = None
                st.success("Crawler data cleared. You can now upload files manually.")
                st.rerun()
        
        # Show processing log if available
        if 'processing_log' in st.session_state and st.session_state.processing_log:
            with st.expander("Processing History"):
                for log_entry in st.session_state.processing_log:
                    st.write(log_entry)
    
    # Only show the regular file upload interface if no crawler data is present
    # or if crawler data was cleared
    if 'pipeline_df' not in st.session_state or st.session_state.pipeline_df is None:
        # Regular file upload interface
        st.write("Upload your data file (CSV, Excel, TSV, etc.)")
        
        uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls", "tsv", "txt"])
        
        if uploaded_file is not None:
            try:
                # Determine file type and read accordingly
                file_extension = uploaded_file.name.split('.')[-1].lower()
                
                if file_extension == 'csv':
                    # Let user specify delimiter
                    delimiter = st.text_input("CSV Delimiter", ",")
                    df = pd.read_csv(uploaded_file, delimiter=delimiter)
                elif file_extension in ['xlsx', 'xls']:
                    # Let user select sheet name
                    xls = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("Select sheet", xls.sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                elif file_extension == 'tsv':
                    df = pd.read_csv(uploaded_file, delimiter='\t')
                else:  # For txt files, let user specify delimiter
                    delimiter = st.text_input("File Delimiter", "\t")
                    df = pd.read_csv(uploaded_file, delimiter=delimiter)
                
                # Display basic info about the data
                st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
                
                # Show a preview of the data
                st.subheader("Data Preview")
                st.dataframe(df.head())
                
                # Store the data in session state
                st.session_state.data = df
                st.session_state.processed_data = df.copy()
                
                # Add to processing log
                if 'processing_log' not in st.session_state:
                    st.session_state.processing_log = []
                
                # Add log entry with timestamp and filename
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.processing_log.append(f"[{timestamp}] Imported {df.shape[0]} rows from file: {uploaded_file.name}")

                # Show data types and missing values
                st.subheader("Data Types and Missing Values")
                data_types = pd.DataFrame(df.dtypes, columns=['Data Type'])
                missing_values = pd.DataFrame(df.isnull().sum(), columns=['Missing Values'])
                data_info = pd.concat([data_types, missing_values], axis=1)
                st.dataframe(data_info)
                
            except Exception as e:
                st.error(f"Error loading file: {e}")
        else:
            st.info("Please upload a file to proceed or use data from the Data Crawler.")
    
    # If data has been saved in session state, show a summary
    if st.session_state.data is not None:
        with st.expander("Current Dataset Summary"):
            st.write(f"Working with dataset: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} columns")
            st.write("First 5 rows:")
            st.dataframe(st.session_state.data.head())

def select_rename_columns():
    """Step 2: Select and rename columns"""
    if st.session_state.data is not None:
        df = st.session_state.data
        
        st.subheader("Original Data Preview")
        st.dataframe(df.head())
        
        # Select columns to keep
        st.subheader("Select Columns")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Select columns to keep", all_columns, default=all_columns)
        
        if selected_columns:
            st.session_state.selected_cols = selected_columns
            # Apply column selection
            filtered_df = df[selected_columns]
            
            # Rename columns
            st.subheader("Rename Columns")
            rename_dict = {}
            
            for col in selected_columns:
                new_name = st.text_input(f"Rename '{col}'", col)
                if new_name != col:
                    rename_dict[col] = new_name
            
            if rename_dict:
                st.session_state.renamed_cols = rename_dict
                filtered_df = filtered_df.rename(columns=rename_dict)
                
                # Show renamed preview
                st.subheader("Preview after renaming")
                st.dataframe(filtered_df.head())
            
            # Update processed data
            st.session_state.processed_data = filtered_df
            
            # Show changes
            st.success(f"Selected {len(selected_columns)} out of {len(all_columns)} columns.")
            
            if rename_dict:
                st.success(f"Renamed {len(rename_dict)} columns.")
        else:
            st.warning("Please select at least one column to continue.")
    else:
        st.error("No data available. Please go back to the upload step.")

def preprocess_data():
    """Step 3: Preprocess data - handle NLP text processing"""
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Process each column based on its type
        st.subheader("Preprocess Text Data")
        
        processed_df = df.copy()
        changes_made = []
        
        # Define text cleaning functions
        def remove_punctuation(text):
            if pd.isna(text):
                return text
            return text.translate(str.maketrans('', '', string.punctuation))

        def remove_numbers(text):
            if pd.isna(text):
                return text
            return re.sub(r'\d+', '', str(text))

        def remove_stopwords(text, language='english'):
            if pd.isna(text):
                return text
            
            if language == 'vietnamese' and underthesea_available:
                # Vietnamese stopwords (simplified list)
                vi_stop_words = ['và', 'của', 'cho', 'trong', 'là', 'với', 'các', 'có', 'được', 'những', 'để', 'không', 'này', 'đã', 'một', 'từ', 'theo', 'về', 'như', 'sau', 'khi']
                tokens = vi_tokenize(str(text))
                return ' '.join(word for word in tokens if word.lower() not in vi_stop_words)
            else:
                stop_words = set(stopwords.words(language))
                return ' '.join(word for word in str(text).split() if word.lower() not in stop_words)

        def remove_spaces(text):
            if pd.isna(text):
                return text
            return ' '.join(str(text).split())

        def remove_emoji(text):
            if pd.isna(text):
                return text
            return emoji.replace_emoji(str(text), replace='')

        def remove_html_tags(text):
            if pd.isna(text):
                return text
            return re.sub(r'<.*?>', '', str(text))

        def remove_urls(text):
            if pd.isna(text):
                return text
            return re.sub(r'http\S+|www\S+|https\S+', '', str(text))

        def convert_to_lowercase(text):
            if pd.isna(text):
                return text
            return str(text).lower()
        
        def tokenize_text(text, language='english'):
            if pd.isna(text):
                return text
            
            if language == 'vietnamese' and underthesea_available:
                return ' '.join(vi_tokenize(str(text)))
            else:
                return ' '.join(word_tokenize(str(text)))
        
        # Process each text column
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        if not text_columns:
            st.info("No text columns found in the dataset.")
        else:
            for column in text_columns:
                st.write(f"**Processing text column: {column}**")
                
                # Language selection for text processing
                language = st.selectbox(
                    f"Select language for '{column}'",
                    ["english", "vietnamese"],
                    key=f"lang_{column}"
                )
                
                # Multi-select for text processing options
                text_options = st.multiselect(
                    f"Select preprocessing steps for '{column}'",
                    [
                        'Remove punctuation',
                        'Remove numbers',
                        'Remove stopwords',
                        'Remove extra spaces',
                        'Remove emoji',
                        'Remove HTML tags',
                        'Remove URLs',
                        'Convert to lowercase',
                        'Tokenize text'
                    ],
                    default=['Remove punctuation', 'Remove stopwords', 'Remove extra spaces'],
                    key=f"nlp_{column}"
                )
                
                # Skip if no options selected
                if not text_options:
                    st.info(f"No preprocessing selected for '{column}'")
                    continue
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Apply selected text processing operations
                total_operations = len(text_options)
                for i, operation in enumerate(text_options):
                    progress_value = (i / total_operations)
                    progress_bar.progress(progress_value)
                    
                    # Make sure the column is string type
                    if not pd.api.types.is_string_dtype(processed_df[column].dtype):
                        processed_df[column] = processed_df[column].astype(str)
                    
                    if operation == 'Remove punctuation':
                        processed_df[column] = processed_df[column].apply(remove_punctuation)
                        changes_made.append(f"Removed punctuation from '{column}'")
                    
                    elif operation == 'Remove numbers':
                        processed_df[column] = processed_df[column].apply(remove_numbers)
                        changes_made.append(f"Removed numbers from '{column}'")
                    
                    elif operation == 'Remove stopwords':
                        processed_df[column] = processed_df[column].apply(lambda x: remove_stopwords(x, language))
                        changes_made.append(f"Removed {language} stopwords from '{column}'")
                    
                    elif operation == 'Remove extra spaces':
                        processed_df[column] = processed_df[column].apply(remove_spaces)
                        changes_made.append(f"Removed extra spaces from '{column}'")
                    
                    elif operation == 'Remove emoji':
                        processed_df[column] = processed_df[column].apply(remove_emoji)
                        changes_made.append(f"Removed emoji from '{column}'")
                    
                    elif operation == 'Remove HTML tags':
                        processed_df[column] = processed_df[column].apply(remove_html_tags)
                        changes_made.append(f"Removed HTML tags from '{column}'")
                    
                    elif operation == 'Remove URLs':
                        processed_df[column] = processed_df[column].apply(remove_urls)
                        changes_made.append(f"Removed URLs from '{column}'")
                    
                    elif operation == 'Convert to lowercase':
                        processed_df[column] = processed_df[column].apply(convert_to_lowercase)
                        changes_made.append(f"Converted '{column}' to lowercase")
                    
                    elif operation == 'Tokenize text':
                        processed_df[column] = processed_df[column].apply(lambda x: tokenize_text(x, language))
                        changes_made.append(f"Tokenized '{column}' using {language} tokenizer")
                
                # Set progress to complete
                progress_bar.progress(1.0)
                
                # Show before/after comparison
                if text_options:
                    comparison_df = pd.DataFrame({
                        'Original': df[column].head(3).tolist(),
                        'Processed': processed_df[column].head(3).tolist()
                    })
                    st.write(f"**Sample results for '{column}':**")
                    st.table(comparison_df)
        
        # For non-text columns (numeric/datetime), provide minimal options
        non_text_columns = [col for col in df.columns if col not in text_columns]
        if non_text_columns:
            st.subheader("Process Non-Text Columns")
            st.write("For numeric and date columns, select operations to apply:")
            
            for column in non_text_columns:
                st.write(f"**Column: {column}** (type: {df[column].dtype})")
                
                # Identify data type
                dtype = df[column].dtype
                
                # For numeric columns - simplified options
                if pd.api.types.is_numeric_dtype(dtype):
                    option = st.selectbox(f"How to handle numeric column '{column}'?", 
                                         ["No change", "Scale (Min-Max)", "Round to nearest"], 
                                         key=f"num_{column}")
                    
                    if option == "Scale (Min-Max)":
                        min_val = df[column].min()
                        max_val = df[column].max()
                        if max_val > min_val:
                            processed_df[column] = (df[column] - min_val) / (max_val - min_val)
                            changes_made.append(f"Applied Min-Max scaling to '{column}'")
                    
                    elif option == "Round to nearest":
                        decimal_places = st.number_input(f"Decimal places for '{column}'", 0, 10, 0, key=f"round_{column}")
                        processed_df[column] = df[column].round(decimal_places)
                        changes_made.append(f"Rounded '{column}' to {decimal_places} decimal places")
                
                # For datetime columns - simplified options
                elif pd.api.types.is_datetime64_dtype(dtype) or column.lower() in ["date", "time", "datetime"]:
                    option = st.selectbox(f"How to handle date column '{column}'?", 
                                         ["No change", "Extract year", "Format date"], 
                                         key=f"date_{column}")
                    
                    # Try to convert to datetime if not already
                    if not pd.api.types.is_datetime64_dtype(dtype):
                        try:
                            processed_df[column] = pd.to_datetime(df[column])
                            changes_made.append(f"Converted '{column}' to datetime format")
                        except:
                            st.warning(f"Could not convert '{column}' to datetime.")
                            continue
                    
                    if option == "Extract year":
                        new_col = f"{column}_year"
                        processed_df[new_col] = processed_df[column].dt.year
                        changes_made.append(f"Extracted year from '{column}' to new column '{new_col}'")
                    
                    elif option == "Format date":
                        date_format = st.text_input(f"Enter date format for {column} (e.g., %Y-%m-%d)", "%Y-%m-%d", key=f"format_{column}")
                        processed_df[column] = processed_df[column].dt.strftime(date_format)
                        changes_made.append(f"Formatted '{column}' with format '{date_format}'")
        
        # Show preview of preprocessed data
        st.subheader("Preview of Preprocessed Data")
        st.dataframe(processed_df.head())
        
        # Show summary of changes
        if changes_made:
            st.subheader("Summary of Preprocessing")
            for change in changes_made:
                st.write(f"✅ {change}")
            
            # Update the processed data
            st.session_state.processed_data = processed_df
            st.session_state.pre_processed = True
        else:
            st.info("No preprocessing changes were applied.")
    else:
        st.error("No data available. Please complete previous steps first.")

def clean_data():
    """Step 4: Clean data - handle missing values and duplicates with simplified options"""
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Display missing values
        missing_values = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing_values.index,
            'Missing Values': missing_values.values,
            'Percentage': (missing_values.values / len(df) * 100).round(2)
        })
        
        st.subheader("Missing Values")
        st.dataframe(missing_df)
        
        # Handle missing values
        st.subheader("Handle Missing Values")
        
        # Create a clean copy of the dataframe
        cleaned_df = df.copy()
        changes_made = []
        
        # Simplified missing values strategy
        if missing_values.sum() > 0:
            st.write("Choose how to handle missing values for each column:")
            
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                
                if missing_count > 0:
                    st.write(f"**Column '{column}' has {missing_count} missing values ({(missing_count/len(df)*100):.2f}%)**")
                    
                    # Very simplified options for handling missing values
                    strategy = st.selectbox(
                        f"Strategy for '{column}'", 
                        ["No change", "Drop rows with missing values", "Replace with empty value"],
                        key=f"missing_{column}"
                    )
                    
                    if strategy == "Drop rows with missing values":
                        original_count = len(cleaned_df)
                        cleaned_df = cleaned_df.dropna(subset=[column])
                        rows_dropped = original_count - len(cleaned_df)
                        changes_made.append(f"Dropped {rows_dropped} rows with missing values in '{column}'")
                    
                    elif strategy == "Replace with empty value":
                        # Different empty values based on column type
                        dtype = df[column].dtype
                        
                        if pd.api.types.is_numeric_dtype(dtype):
                            # For numeric columns, replace with 0
                            cleaned_df[column] = cleaned_df[column].fillna(0)
                            changes_made.append(f"Filled {missing_count} missing values in '{column}' with 0")
                        
                        elif pd.api.types.is_datetime64_dtype(dtype):
                            # For datetime, we'll use a far-past date as empty value
                            cleaned_df[column] = cleaned_df[column].fillna(pd.Timestamp('1900-01-01'))
                            changes_made.append(f"Filled {missing_count} missing values in '{column}' with 1900-01-01")
                        
                        else:
                            # For strings and other types, use empty string
                            cleaned_df[column] = cleaned_df[column].fillna("")
                            changes_made.append(f"Filled {missing_count} missing values in '{column}' with empty string")
        else:
            st.info("No missing values found in the dataset.")
        
        # Handle duplicates
        st.subheader("Handle Duplicate Rows")
        
        # Check for duplicates
        duplicates = cleaned_df.duplicated().sum()
        
        if duplicates > 0:
            st.warning(f"Found {duplicates} duplicate rows in the dataset.")
            
            dup_strategy = st.selectbox("How to handle duplicates?", ["Keep all", "Drop duplicates"])
            
            if dup_strategy == "Drop duplicates":
                # Let user choose which columns to consider for determining duplicates
                dup_subset = st.multiselect("Consider only these columns for duplicates (empty = all columns)", 
                                           cleaned_df.columns.tolist())
                
                original_shape = cleaned_df.shape
                if dup_subset:
                    cleaned_df = cleaned_df.drop_duplicates(subset=dup_subset)
                    subset_str = ", ".join(dup_subset)
                    changes_made.append(f"Dropped duplicates based on columns: {subset_str}")
                else:
                    cleaned_df = cleaned_df.drop_duplicates()
                    changes_made.append("Dropped all duplicate rows")
                
                new_shape = cleaned_df.shape
                rows_removed = original_shape[0] - new_shape[0]
                
                if rows_removed > 0:
                    st.success(f"Removed {rows_removed} duplicate rows.")
        else:
            st.info("No duplicate rows found in the dataset.")
        
        # Show preview of cleaned data
        st.subheader("Preview of Cleaned Data")
        st.dataframe(cleaned_df.head())
        
        # Show data shape
        st.write(f"Data shape: {cleaned_df.shape[0]} rows, {cleaned_df.shape[1]} columns")
        
        # Show summary of changes
        if changes_made:
            st.subheader("Summary of Cleaning")
            for change in changes_made:
                st.write(f"✅ {change}")
            
            # Update the processed data
            st.session_state.processed_data = cleaned_df
            st.session_state.cleaned = True
        else:
            st.info("No cleaning changes were applied.")
    else:
        st.error("No data available. Please complete previous steps first.")

def export_data():
    """Step 5: Export processed and cleaned data"""
    if st.session_state.processed_data is not None:
        df = st.session_state.processed_data
        
        st.subheader("Final Data Preview")
        st.dataframe(df.head())
        
        # Data info
        st.write(f"Final data shape: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Export options
        st.subheader("Export Data")
        
        export_format = st.selectbox("Choose export format", ["CSV", "Excel", "JSON", "Pickle"])
        
        # Additional export options based on format
        if export_format == "CSV":
            delimiter = st.selectbox("Delimiter", [",", ";", "\\t"], 0)
            delimiter = "\t" if delimiter == "\\t" else delimiter
            
            # Generate the file for download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, sep=delimiter)
            
            # Create download button
            today = datetime.now().strftime("%Y%m%d")
            filename = f"processed_data_{today}.csv"
            
            st.download_button(
                label="Download CSV",
                data=csv_buffer.getvalue(),
                file_name=filename,
                mime="text/csv"
            )
            
            st.success(f"Your data is ready for download as {filename}!")
        
        elif export_format == "Excel":
            # Generate Excel file for download
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name="Processed Data")
            
            # Create download button
            today = datetime.now().strftime("%Y%m%d")
            filename = f"processed_data_{today}.xlsx"
            
            st.download_button(
                label="Download Excel",
                data=excel_buffer.getvalue(),
                file_name=filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            
            st.success(f"Your data is ready for download as {filename}!")
        
        elif export_format == "JSON":
            # Generate JSON file for download
            json_str = df.to_json(orient="records")
            
            # Create download button
            today = datetime.now().strftime("%Y%m%d")
            filename = f"processed_data_{today}.json"
            
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=filename,
                mime="application/json"
            )
            
            st.success(f"Your data is ready for download as {filename}!")
        
        elif export_format == "Pickle":
            # Generate Pickle file for download
            pickle_buffer = io.BytesIO()
            df.to_pickle(pickle_buffer)
            
            # Create download button
            today = datetime.now().strftime("%Y%m%d")
            filename = f"processed_data_{today}.pkl"
            
            st.download_button(
                label="Download Pickle",
                data=pickle_buffer.getvalue(),
                file_name=filename,
                mime="application/octet-stream"
            )
            
            st.success(f"Your data is ready for download as {filename}!")
        
        # Reset pipeline option
        st.write("---")
        if st.button("Start New Pipeline"):
            # Reset all session state
            for key in ['data', 'step', 'processed_data', 'selected_cols', 'renamed_cols', 'pre_processed', 'cleaned']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
    else:
        st.error("No processed data available. Please complete previous steps first.")

if __name__ == "__main__":
    app()

