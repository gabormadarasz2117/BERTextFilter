import os
import re
from collections import defaultdict

def find_duplicate_sentences(folder_path):
    # Dictionary to store sentences and their occurrences
    sentence_dict = defaultdict(list)
    
    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line_number, line in enumerate(lines):
                    # Split the line into sentences
                    sentences = re.split(r'(?<=[.!?]) +', line)
                    for sentence in sentences:
                        cleaned_sentence = sentence.strip()
                        if cleaned_sentence:
                            sentence_dict[cleaned_sentence].append((filename, line_number, line))
    
    # Find duplicate sentences
    duplicate_sentences = {sentence: occurrences for sentence, occurrences in sentence_dict.items() if len(occurrences) > 1}
    
    return duplicate_sentences

def print_duplicates(duplicate_sentences):
    for sentence, occurrences in duplicate_sentences.items():
        print(f'Duplicate sentence: "{sentence}"')
        for occurrence in occurrences:
            print(f'Found in file {occurrence[0]} at line {occurrence[1]}')
        print()

def delete_duplicates(folder_path, duplicate_sentences):
    for sentence, occurrences in duplicate_sentences.items():
        for occurrence in occurrences:
            filename, line_number, line = occurrence
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            with open(filepath, 'w', encoding='utf-8') as file:
                for i, l in enumerate(lines):
                    if i == line_number:
                        # Remove the sentence from the line
                        new_line = re.sub(re.escape(sentence), '', l)
                        file.write(new_line)
                    else:
                        file.write(l)
    print("Duplicate sentences have been deleted.")

def main(folder_path, user_input):
    duplicate_sentences = find_duplicate_sentences(folder_path)
    if duplicate_sentences:
        print_duplicates(duplicate_sentences)
        #user_input = input("Do you want to delete the duplicate sentences from the files? (yes/no): ").strip().lower()
        if user_input in ["yes", "y"]:
            delete_duplicates(folder_path, duplicate_sentences)
        else:
            print("No changes were made to the files.")
    else:
        print("No duplicate sentences found.")

if __name__ == "__main__":
    folder_path = './cleaned_meteor_text/'
    main(folder_path)
