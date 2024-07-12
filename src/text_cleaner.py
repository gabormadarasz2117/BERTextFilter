import os
import re
import math
import spacy
import huspacy
import pandas as pd
import numpy as np
import torch

from datasets import Dataset
from tqdm import tqdm
from quntoken import tokenize
from transformers import pipeline
from transformers.pipelines.pt_utils import KeyDataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.signal import argrelextrema

#models
gpus = [i for i in range(torch.cuda.device_count())]
print("Available GPUIDs:", gpus)
device = int(input("Please enter the GPUID to use! (-1 to CPU): ").strip())
pipe = pipeline("text-classification", model="NYTK/hucola-puli-bert-large-hungarian", max_length=512, truncation=True, device=device, batch_size=32)
model = SentenceTransformer('NYTK/sentence-transformers-experimental-hubert-hungarian').to(f"cuda:{device}")

# Function to clean text
def clean_text(text_list):
    cleaned_text_list = []
    for text in text_list:
        # Remove page numbers and excessive whitespace
        text = re.sub(r"\f", " ", text)
        text = re.sub(r"\s{2,}", " ", text)
        
        # Remove image captions (assuming they're within parentheses or marked)
        text = re.sub(r"\(.*?\)", "", text)
        
        # Remove very short lines and single words (titles, headings)
        cleaned_lines = []
        for line in text.split('\n'):
            # Remove lines with more than 3 consecutive dots
            if '. . . .' in line:
                continue
            if len(line.split()) > 3:  # keep lines with more than 3 words
                cleaned_lines.append(line.strip())
        
        # Join the cleaned lines
        cleaned_text = " ".join(cleaned_lines)
        
        # Remove email addresses
        cleaned_text = re.sub(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', cleaned_text)
        
        # Remove remaining unwanted short sections
        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text)  # normalize whitespace
        
        # Add the cleaned text to the list
        cleaned_text_list.append(cleaned_text.strip())
    
        #Remove only UPPERCASE rows
        cleaned_text_list = [s for s in cleaned_text_list if not s.isupper()]

    return cleaned_text_list
    
def character_ratios(text):
    letters = sum(char.isalpha() for char in text)
    numbers = sum(char.isdigit() for char in text)
    total_characters = letters + numbers
    return numbers / total_characters if total_characters else 1

def check_hungarian_chars(text):
    hungarian_chars = set("aábcdeéfghiíjklmnoóöőpqrstuúüűvwxyz"
                          "AÁBCDEÉFGHIÍJKLMNOÓÖŐPQRSTUÚÜŰVWXYZ"
                          ".,!?-;:()[]{}<>@#$%^&*+=/\\_|~`'\" ")
    return text if all(char in hungarian_chars for char in text) else None

def convert_to_standard_hungarian(text):
    char_map = {'á': 'á', 'ě': 'é', 'í': 'í', 'ó': 'ó', 'ö': 'ö', 'ô': 'ő',
                'ú': 'ú', 'ü': 'ü', 'ű': 'ű', 'Á': 'Á', 'É': 'É', 'Í': 'Í',
                'Ó': 'Ó', 'Ö': 'Ö', 'Ő': 'Ő', 'Ú': 'Ú', 'Ü': 'Ü', 'Ű': 'Ű',
                'û': 'ű', 'Û': 'Ű'}
    return ''.join(char_map.get(char, char) for char in text)

def remove_space_before_punctuation(text):
    punctuation_marks = [".", ",", ";", ":", "!", "?"]
    for mark in punctuation_marks:
        text = text.replace(f" {mark}", mark)
    return text

def remove_hyphenation(lines):
    """
    Eltávolítja a sorvégi elválasztójeleket, és összevonja az elválasztott szavakat egy szöveges fájlban.
    """
    cleaned_lines = []
    buffer = ""

    for line in lines:
        stripped_line = line.rstrip()  # Sorvégi whitespace eltávolítása
        if stripped_line.endswith('-'):
            buffer += stripped_line[:-1]  # Elválasztójel eltávolítása és sor vége bufferbe
        else:
            buffer += stripped_line  # Hozzáadjuk a sor végét a bufferhez
            cleaned_lines.append(buffer)
            buffer = ""  # Buffer ürítése a következő sorhoz

    # Maradék buffer hozzáadása, ha van
    if buffer:
        cleaned_lines.append(buffer)

    return "\n".join(cleaned_lines)

def rev_sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(0.5 * x))

def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    """Function returns list of weighted sums of activated sentence similarities.

    Args:
        similarities (numpy array): it should be a square matrix where each sentence corresponds to another with cosine similarity.
        p_size (int): number of sentences used to calculate weighted sum.

    Returns:
        list: list of weighted sums.
    """
    num_sentences = similarities.shape[0]
    
    # If the number of sentences is less than p_size, adjust p_size
    if (num_sentences < p_size):
        p_size = num_sentences
    
    # Create the space for weights
    x = np.linspace(-10, 10, p_size)
    
    # Apply the activation function
    y = np.vectorize(rev_sigmoid)
    activation_weights = y(x)
    
    # If necessary, pad the activation_weights
    if (num_sentences > p_size):
        activation_weights = np.pad(activation_weights, (0, num_sentences - p_size))
    
    # Calculate diagonals and apply weights
    diagonals = [similarities.diagonal(each) for each in range(num_sentences)]
    diagonals = [np.pad(each, (0, num_sentences - len(each))) for each in diagonals]
    diagonals = np.stack(diagonals)
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    
    # Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    
    return activated_similarities


def process_file(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    if not lines:
        print(f"The file {input_file_path} is empty.")
        return

    text = remove_hyphenation(lines)
    
    tokenized_list = list(tokenize(text, mode='sentence', word_break=True, form='spl'))
    
    text = clean_text(tokenized_list)
    
    
    # Szövegek dictionary-k listájának létrehozása
    szoveg_dict_lista = [{"text": szoveg} for szoveg in text]

    # Dataset létrehozása
    dataset = Dataset.from_list(szoveg_dict_lista)


    # pipe használata
    checked = []
    for original_text, result in tqdm(zip(dataset["text"], pipe(KeyDataset(dataset, "text"))), total=len(dataset), desc="Processing sentences"):
        entry = {"text": original_text}
        entry["label"] = result["label"]
        checked.append(entry)

# Az eredmény lista létrehozása
    #checked = processed_dataset.to_list()


    filtered_list = [item["text"] for item in checked if item["label"] == "LABEL_1"]
    email_filtered_list = [item for item in filtered_list if not re.search(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', item)]

    char_filtered_list = [item for item in email_filtered_list if character_ratios(item) < 0.2]
    accent_fixed = [convert_to_standard_hungarian(item).replace("\n", "") for item in char_filtered_list]
    hun_char_filtered = [item for item in accent_fixed if check_hungarian_chars(item)]
    punc_fixed = [remove_space_before_punctuation(item) for item in hun_char_filtered]
    whitespace_fixed = [s.lstrip() for s in punc_fixed]
    final_text = " ".join(whitespace_fixed)

    if not final_text.strip():
        print("The final text is empty. No sentences were generated.")
        return

    nlp = spacy.load("hu_core_news_md")
    nlp.max_length = 9000000
    doc = nlp(final_text, disable=['ner', 'parser', 'morphologizer', 'lookup_lemmatizer', 'trainable_lemmatizer'])
    sentences = list(sent.text for sent in doc.sents)

    if not sentences:
        print("No sentences were found in the text.")
        return

    try:
        
        embeddings = model.encode(sentences)
    except Exception as e:
        print(f"Error encoding sentences: {e}")
        return

    similarities = cosine_similarity(embeddings)
    activated_similarities = activate_similarities(similarities, p_size=5)
    minmimas = argrelextrema(activated_similarities, np.less, order=3)
    split_points = [each for each in minmimas[0]]

    text = ''
    for num, each in enumerate(sentences):
        if num in split_points:
            text += f'\n{each} '
        else:
            text += f'{each} '

    with open(output_file_path, 'w', encoding='utf-8') as file:
        file.write(text)

    print(f"Cleaned text saved to {output_file_path}")

def process_all_files(input_folder_path, output_folder_path):
    for filename in os.listdir(input_folder_path):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder_path, filename)
            output_file_path = os.path.join(output_folder_path, f"cleaned_{filename}")
            process_file(input_file_path, output_file_path)
