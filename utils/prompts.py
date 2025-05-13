from string import Template
from langchain.prompts  import PromptTemplate   
from langchain_core.prompts import ChatPromptTemplate


LABEL_CLASSIFY = PromptTemplate(
    input_variables=["query", "unique_labels"],
    template="""
        Classify the following query into one of the unique labels with the given description.  
        Query: {query}
        Unique Labels: {unique_labels}
        Note: Just output the label name (str), do not include any other text. [`label`: value, `description`: description]. You will return value
        
        """,
)
