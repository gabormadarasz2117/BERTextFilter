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

def delete_small_txt_files(folder_path, size_threshold=10*1024):
    """
    Törli az összes 10KB-nál kisebb .txt fájlt a megadott mappából.

    :param folder_path: A mappa elérési útja, ahol a fájlokat ellenőrizni kell.
    :param size_threshold: A méretküszöb, amely alatt a fájlokat törölni kell (alapértelmezett 10KB).
    """
    # Bejárjuk a mappát
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ellenőrizzük, hogy a fájl egy .txt fájl-e és hogy kisebb-e a küszöbnél
        if filename.endswith(".txt") and os.path.isfile(file_path) and os.path.getsize(file_path) < size_threshold:
            try:
                os.remove(file_path)
                print(f"Törölt fájl: {file_path}")
            except Exception as e:
                print(f"Hiba történt a törlés során: {file_path}. Hiba: {e}")

def main(folder_path, dataset_name, remove_small_files):
    """
    Main function to read .txt files from a folder, create a Hugging Face Dataset,
    split it into train, validation, and test sets, and save them to disk.

    Args:
    - folder_path (str): Path to the folder containing .txt files.

    Returns:
    - dataset_dict (DatasetDict): Dictionary containing 'train', 'validation', and 'test' datasets.
    """
    if remove_small_files in ["yes", "y"]:
        delete_small_txt_files(folder_path)
    else:
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
    remove_small_files = input("Delete small txt files? < 10KB (yes / no)")
    dataset = main(folder_path, dataset_name, remove_small_files)
    
    # Example to print the dataset
    print(dataset)
