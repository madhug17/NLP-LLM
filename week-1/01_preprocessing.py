import re 
import string
text = """
The students are learning advanced machine learning.
They were studying different algorithms and models.
The models are performing exceptionally well!
"""
text = text.lower()
text = re.sub(r'https?://\S+|www\.\S+', '', text)
text = text.translate(str.maketrans('','',string.punctuation))
text = re.sub(r'\s+', ' ', text).strip()
print("Processed Text:")
print(text)