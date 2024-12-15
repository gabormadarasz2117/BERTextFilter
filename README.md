# BERTextFilter

This project provides a Hungarian text processing pipeline that includes filtering based on the grammatical correctness of text files, handling duplicate sentences, and optionally creating a Hugging Face dataset.
It uses two language models to accompish the task: 
-NYTK/hucola-puli-bert-large-hungarian 
-NYTK/sentence-transformers-experimental-hubert-hungarian



# Install

    Clone:
        git clone https://git.nlp.nytud.hu/madaraszg/BERTextFilter.git
        
    Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    
# Usage

Main script:

    ```bash
    python3 bertextfilter.py
    ```



## The main script performs the following steps:

    Prompts the user to provide the GPU ID to be used.

    Prompts the user to specify the input folder path and the output folder path.

    Checks if the input folder exists.

    Creates the output folder if it does not exist.

    Processes the text files in the input folder using the process_all_files function from text_cleaner.py.

    Identifies and lists duplicate sentences in the processed files using the main function from deduplicate.py.

    Asks the user whether they want to delete the duplicate sentences.

    Asks the user whether they want to create a Hugging Face dataset:
        If yes, prompts for the dataset name and creates the dataset using the main function from dataset_creation.py.
