import os
from datasets import Dataset, DatasetDict

def read_txt_files_from_folder(folder_path):
    """
    Reads all .txt files from a folder and returns a list of dictionaries,
    where each dictionary contains a single text entry.

    Args:
    - folder_path (str): Path to the folder containing .txt files.

    Returns:
    - data (list): List of dictionaries, each containing a "text" entry.
    """
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                for line in lines:
                    data.append({"text": line.strip()})
    return data

def create_huggingface_dataset(data):
    """
    Creates a Hugging Face Dataset object from a list of dictionaries.

    Args:
    - data (list): List of dictionaries, where each dictionary contains a "text" entry.

    Returns:
    - dataset (Dataset): Hugging Face Dataset object.
    """
    dataset = Dataset.from_list(data)
    return dataset

def main(folder_path, dataset_name):
    """
    Main function to read .txt files from a folder, create a Hugging Face Dataset,
    split it into train, validation, and test sets, and save them to disk.

    Args:
    - folder_path (str): Path to the folder containing .txt files.

    Returns:
    - dataset_dict (DatasetDict): Dictionary containing 'train', 'validation', and 'test' datasets.
    """
    data = read_txt_files_from_folder(folder_path)
    dataset = create_huggingface_dataset(data)
    
    # Split the dataset into train, validation, and test sets
    train_testvalid = dataset.train_test_split(test_size=0.1)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
    
    dataset_dict = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['test'],
        'test': test_valid['train']
    })
    
    # Save the dataset to disk (optional)
    dataset_dict.save_to_disk(dataset_name)
    
    return dataset_dict

# This block allows the script to be executed standalone
if __name__ == "__main__":
    folder_path = input("Please enter the input folder path: ")
    dataset_name = input("Please enter the dataset's name: ")
    dataset = main(folder_path, dataset_name)
    
    # Example to print the dataset
    print(dataset)
