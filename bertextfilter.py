import os
from src.text_cleaner import process_all_files
from src.deduplicate import main as find_duplicates_main
from src.dataset_creation import main as create_huggingface_dataset

def main():
    input_folder = input("Please enter the input folder path: ").strip()
    output_folder = input("Please enter the output folder path: ").strip()
    
    # Ask the user if they want to create a Hugging Face dataset
    create_dataset = input("Do you want to create a HuggingFace dataset? (yes/no): ").strip().lower()
    
    if create_dataset in ["yes", "y"]:
        dataset_name = input("Please enter the dataset's name: ").strip().lower()
        remove_small_files = input("Delete small txt files from dataset? < 10KB (yes / no)").strip().lower()
    
    # Ask the user if they want to deduplacate
    deduplicate = input("Do you want to delete duplacated sentences? (yes/no): ").strip().lower()
    
    if not os.path.exists(input_folder):
        print(f"Input folder '{input_folder}' does not exist.")
        return
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Output folder '{output_folder}' created.")

    # Call the text cleaning process
    print("Processing files...")
    process_all_files(input_folder, output_folder)
    print("Files processed.")

    # Call the duplicate sentence handling
    if deduplicate in ["yes", "y"]:
        print("Finding and handling duplicate sentences...")
        find_duplicates_main(output_folder, deduplicate)
        print("Duplicate sentence handling complete.")

    # Create HF dataset
    if create_dataset in ["yes", "y"]:
        print("Creating Hugging Face dataset...")
        create_huggingface_dataset(output_folder, dataset_name, remove_small_files)
        print("Hugging Face dataset creation complete.")

if __name__ == "__main__":
    main()
