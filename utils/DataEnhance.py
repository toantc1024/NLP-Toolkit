# enum 

import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from utils.prompts import LABEL_CLASSIFY
from langchain_google_genai import GoogleGenerativeAI
from enum import Enum
from pydantic import Field
from pydantic import BaseModel
from utils.Embedding import hf
from config import settings
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.flow as naf
import pandas as pd
import numpy as np

class UniqueLabel(BaseModel):
    label: str
    description: str
    
class LLMLabelClassifierOutput(BaseModel):
    label: str = Field(description="The label predicted")
    

class DataEnhanceType(Enum):
    SYNONYM_AUGMENT = 1
    RANDOM_DELETION = 2
    ANTONYM_AUGMENT = 3
    SPELLING_AUGMENT = 4
    CONTEXTUAL_AUGMENT = 5

# Dictionary mapping enhancement types to human-readable names
ENHANCE_TYPE_NAMES = {
    DataEnhanceType.SYNONYM_AUGMENT: "Synonym Replacement",
    DataEnhanceType.RANDOM_DELETION: "Random Word Deletion",
    DataEnhanceType.ANTONYM_AUGMENT: "Antonym Replacement",
    DataEnhanceType.SPELLING_AUGMENT: "Spelling Augmentation",
    DataEnhanceType.CONTEXTUAL_AUGMENT: "Contextual Word Embedding"
}

def enhance_text(text, enhance_type: DataEnhanceType, num_samples: int = 1):
    if enhance_type == DataEnhanceType.SYNONYM_AUGMENT:
        aug = naw.SynonymAug(aug_src='wordnet')
    elif enhance_type == DataEnhanceType.RANDOM_DELETION:
        aug = naw.RandomWordAug(action="delete")
    elif enhance_type == DataEnhanceType.ANTONYM_AUGMENT:
        aug = naw.AntonymAug()
    elif enhance_type == DataEnhanceType.SPELLING_AUGMENT:
        aug = nac.KeyboardAug(aug_char_min=1, aug_char_max=2)
    elif enhance_type == DataEnhanceType.CONTEXTUAL_AUGMENT:
        aug = naw.ContextualWordEmbsAug(model_path='distilbert-base-uncased', action="substitute")
    
    enhanced_samples = []
    try:
        for _ in range(num_samples):
            augmented_text = aug.augment(text)
            if isinstance(augmented_text, list):
                enhanced_samples.extend(augmented_text)
            else:
                enhanced_samples.append(augmented_text)
    except Exception as e:
        # If augmentation fails, return the original text
        enhanced_samples = [f"Error: {str(e)}. Original text: {text}"]
        
    return enhanced_samples

def enhance_dataframe(df, text_column, label_column=None, enhance_type=DataEnhanceType.SYNONYM_AUGMENT, 
                     samples_per_text=1, unique_labels=None, remove_duplicates=True):
    enhanced_rows = []
    # Create a set to track unique texts
    seen_texts = set()
    
    # Add original texts to the set
    for _, row in df.iterrows():
        seen_texts.add(row[text_column])
    
    for idx, row in df.iterrows():
        original_text = row[text_column]
        current_label = None
        
        if label_column and unique_labels:
            current_label = UniqueLabel(label=row[label_column], description="")
            
        # Generate enhanced texts
        enhanced_texts = enhance_text(original_text, enhance_type, samples_per_text)
        
        for enhanced_text in enhanced_texts:
            # Skip if this is a duplicate and remove_duplicates is True
            if remove_duplicates and enhanced_text in seen_texts:
                continue
                
            # Add to tracking set
            seen_texts.add(enhanced_text)
            
            new_row = row.copy()
            new_row[text_column] = enhanced_text
            
            # Predict new label if label column and unique labels are provided
            if label_column and unique_labels and current_label:
                predicted_label = predict_label(original_text, enhanced_text, unique_labels, current_label)
                new_row[label_column] = predicted_label
                
            enhanced_rows.append(new_row)
    
    enhanced_df = pd.DataFrame(enhanced_rows)
    return enhanced_df

def predict_label(original_text, augmented_text, unique_labels: list[UniqueLabel], current_label: UniqueLabel = None):
    # If descriptions are provided, use them for better label prediction
    if any(label.description for label in unique_labels):
        # Format labels with descriptions for the LLM
        text = ""
        
        for label in unique_labels:
            text += f"[`label`: {label.label}, `description`: {label.description}],\n"
        
        # Check similarity between original and augmented text
        if original_text and current_label:
            original_embedding = [hf.embed_query(original_text)]
            augmented_embedding = [hf.embed_query(augmented_text)]
            similarity = cosine_similarity(original_embedding, augmented_embedding)
            
            # If very similar, keep the original label
            if similarity > 0.95:
                return current_label.label
        
        # For significant changes or no original text, use LLM prediction
        llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=settings.GOOGLE_GEN_AI_API_KEY)
        prompt = LABEL_CLASSIFY.format(query=augmented_text, unique_labels=text)
        
        response = llm(prompt)
        return response.strip()
    else:
        # Fallback to similarity-based prediction if no descriptions
        if original_text and current_label:
            original_embedding = [hf.embed_query(original_text)]
            augmented_embedding = [hf.embed_query(augmented_text)]
            similarity = cosine_similarity(original_embedding, augmented_embedding)
            
            if similarity > 0.90:
                return current_label.label
            
        # Use simple prompt without descriptions
        text = ", ".join([label.label for label in unique_labels])
        llm = GoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=settings.GOOGLE_GEN_AI_API_KEY)
        prompt = f"Classify the following text into one of these labels: {text}.\n\nText: {augmented_text}\n\nLabel:"
        
        response = llm(prompt)
        return response.strip()
