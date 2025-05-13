import streamlit as st
import pandas as pd
import io
from utils.DataEnhance import DataEnhanceType, enhance_dataframe, ENHANCE_TYPE_NAMES, UniqueLabel

def app():
    st.title("Data Augmentation")
    
    # Step 1: Upload a file
    st.header("Step 1: Upload your data")
    uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.success(f"File '{uploaded_file.name}' loaded successfully with {len(df)} rows and {len(df.columns)} columns.")
            
            # Step 2: Select columns for enhancement
            st.header("Step 2: Select columns")
            text_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            if not text_columns:
                st.error("No text columns found in the dataset.")
                return
            
            text_column = st.selectbox(
                "Select the text column to enhance:", 
                options=text_columns
            )
            
            use_label = st.checkbox("Predict labels after enhancement", value=False)
            label_column = None
            unique_labels = None
            
            if use_label:
                all_columns = df.columns.tolist()
                label_column = st.selectbox(
                    "Select the label column:", 
                    options=all_columns
                )
                
                # Get unique labels
                unique_label_values = df[label_column].unique()
                
                # Allow users to provide descriptions for each label
                st.subheader("Provide descriptions for each label")
                st.info("Descriptions help the system better understand each label's meaning for more accurate predictions.")
                
                unique_labels = []
                
                # Create a form for label descriptions
                with st.form(key='label_descriptions_form'):
                    label_descriptions = {}
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    for i, label_value in enumerate(unique_label_values):
                        # Alternate between columns for better space usage
                        with col1 if i % 2 == 0 else col2:
                            st.subheader(f"Label: {label_value}")
                            description = st.text_area(
                                f"Description for '{label_value}'",
                                key=f"desc_{label_value}",
                                help="What does this label mean? Provide context and examples.",
                                placeholder=f"e.g., '{label_value}' represents..."
                            )
                            label_descriptions[label_value] = description
                    
                    submit_button = st.form_submit_button(label='Save Descriptions')
                    
                    if submit_button:
                        st.success("Label descriptions saved!")
                
                # Convert to UniqueLabel objects
                for label_value, description in label_descriptions.items():
                    unique_labels.append(UniqueLabel(label=str(label_value), description=description))
            
            # Step 3: Choose augmentation settings
            st.header("Step 3: Choose enhancement settings")
            enhance_type = st.selectbox(
                "Select enhancement type:",
                options=list(DataEnhanceType),
                format_func=lambda x: ENHANCE_TYPE_NAMES.get(x, str(x))
            )
            
            # Display warning for antonym augmentation
            
            samples_per_text = st.slider(
                "Number of augmented samples per text:", 
                min_value=1, 
                max_value=5, 
                value=1
            )
            
            # Option to remove duplicates
            remove_duplicates = st.checkbox("Remove duplicate enhancements", value=True,
                help="Prevent identical texts from appearing in the enhanced dataset")
            
            # Preview original data
            st.subheader("Original Data Preview")
            st.dataframe(df.head())
            
            # Step 4: Preview enhancement
            st.header("Step 4: Preview enhancement")
            if st.button("Generate Preview"):
                if use_label and not unique_labels:
                    st.error("Please provide descriptions for your labels before generating a preview.")
                else:
                    with st.spinner("Generating preview..."):
                        # Get a sample of 2 rows for preview
                        sample_df = df.sample(min(2, len(df)))
                        
                        # Enhance the sample
                        enhanced_sample = enhance_dataframe(
                            sample_df, 
                            text_column, 
                            label_column, 
                            enhance_type, 
                            samples_per_text, 
                            unique_labels,
                            remove_duplicates
                        )
                        
                        # Show a comparison with original and enhanced text
                        st.subheader("Enhanced Data Preview")
                        
                        with st.expander("View Comparison", expanded=True):
                            for _, orig_row in sample_df.iterrows():
                                st.markdown("---")
                                st.markdown("**Original Text:**")
                                st.markdown(f"> {orig_row[text_column]}")
                                
                                if label_column:
                                    st.markdown(f"**Original Label:** {orig_row[label_column]}")
                                
                                # Find corresponding enhanced rows
                                enhanced_rows = enhanced_sample[enhanced_sample.index == orig_row.name]
                                
                                for i, (_, enh_row) in enumerate(enhanced_rows.iterrows()):
                                    st.markdown(f"**Enhanced Text {i+1}:**")
                                    st.markdown(f"> {enh_row[text_column]}")
                                    
                                    if label_column:
                                        st.markdown(f"**Enhanced Label:** {enh_row[label_column]}")
                        
                        st.dataframe(enhanced_sample)
            
            # Step 5: Apply enhancement to full dataset
            st.header("Step 5: Enhance full dataset")
            if st.button("Enhance Full Dataset"):
                if use_label and not unique_labels:
                    st.error("Please provide descriptions for your labels before enhancing the dataset.")
                else:
                    with st.spinner("Enhancing data... This may take a while for large datasets."):
                        # Enhance the full dataset
                        enhanced_df = enhance_dataframe(
                            df, 
                            text_column, 
                            label_column, 
                            enhance_type, 
                            samples_per_text, 
                            unique_labels,
                            remove_duplicates
                        )
                        
                        # Store enhanced dataframe in session state for later use
                        st.session_state.enhanced_df = enhanced_df
                        st.session_state.original_df = df
                        
                        # If no duplicates are removed, show a special success message
                        if remove_duplicates:
                            original_size = len(df) * (samples_per_text + 1)  # Expected size without deduplication
                            actual_size = len(enhanced_df) + len(df)  # Actual size with deduplication + original
                            duplicates_removed = original_size - actual_size
                            
                            if duplicates_removed > 0:
                                st.success(f"Data enhanced successfully! New dataset has {len(enhanced_df)} rows. " +
                                          f"Removed {duplicates_removed} duplicate enhancements.")
                            else:
                                st.success(f"Data enhanced successfully! New dataset has {len(enhanced_df)} rows. " +
                                          f"No duplicate enhancements were found.")
                        else:
                            st.success(f"Data enhanced successfully! New dataset has {len(enhanced_df)} rows.")
                        
                        # Display a preview of the enhanced dataset
                        st.subheader("Enhanced Dataset Preview")
                        st.dataframe(enhanced_df.head(10))
                        
                        # Step 6: Export options
                        st.header("Step 6: Export options")
                        
                        # Create two columns for export options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Export Enhanced Dataset Only")
                            export_format_enhanced = st.selectbox(
                                "Select export format for enhanced data:",
                                options=["CSV", "Excel"],
                                key="export_format_enhanced"
                            )
                            
                            if export_format_enhanced == "CSV":
                                csv = enhanced_df.to_csv(index=False)
                                st.download_button(
                                    label="Download Enhanced Data (CSV)",
                                    data=csv,
                                    file_name="enhanced_data.csv",
                                    mime="text/csv"
                                )
                            else:
                                # Excel export
                                output = io.BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    enhanced_df.to_excel(writer, index=False, sheet_name='Enhanced Data')
                                excel_data = output.getvalue()
                                st.download_button(
                                    label="Download Enhanced Data (Excel)",
                                    data=excel_data,
                                    file_name="enhanced_data.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                        
                        with col2:
                            st.subheader("Merge with Original Dataset")
                            
                            merge_with_original = st.checkbox(
                                "Include original data in export", 
                                value=True,
                                help="Combine the original dataset with the enhanced data"
                            )
                            
                            if merge_with_original:
                                # Create a combined dataset
                                combined_df = pd.concat([df, enhanced_df], ignore_index=True)
                                
                                # Add a column to identify original vs enhanced rows
                                add_source_column = st.checkbox(
                                    "Add source identifier column", 
                                    value=True,
                                    help="Adds a column that identifies which rows are original and which are enhanced"
                                )
                                
                                if add_source_column:
                                    source_column_name = st.text_input(
                                        "Source column name",
                                        value="data_source",
                                        help="Name of the column that will identify original vs enhanced data"
                                    )
                                    
                                    # Create a new column with source information
                                    combined_df[source_column_name] = "enhanced"
                                    combined_df.loc[0:len(df)-1, source_column_name] = "original"
                                
                                # Preview the combined dataset
                                st.subheader("Combined Dataset Preview")
                                st.dataframe(combined_df.head(10))
                                
                                # Export options for combined dataset
                                export_format_combined = st.selectbox(
                                    "Select export format for combined data:",
                                    options=["CSV", "Excel"],
                                    key="export_format_combined"
                                )
                                
                                if export_format_combined == "CSV":
                                    csv = combined_df.to_csv(index=False)
                                    st.download_button(
                                        label="Download Combined Data (CSV)",
                                        data=csv,
                                        file_name="combined_data.csv",
                                        mime="text/csv"
                                    )
                                else:
                                    # Excel export
                                    output = io.BytesIO()
                                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                        combined_df.to_excel(writer, index=False, sheet_name='Combined Data')
                                    excel_data = output.getvalue()
                                    st.download_button(
                                        label="Download Combined Data (Excel)",
                                        data=excel_data,
                                        file_name="combined_data.xlsx",
                                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                    )
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    