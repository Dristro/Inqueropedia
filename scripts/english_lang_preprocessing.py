### Language detection
from langdetect import detect, LangDetectException
from multiprocessing import Pool
from pathlib import Path
import pandas as pd
import numpy as np
import re
import os

def is_valid_line(line: str) -> bool:
    """
    Quickly checks if the line is valid without detecting the language.

    Args:
        line: line to check on
    Returns:
        True if line is valid
        False if otherwise
        None if more processing is requeried
    """
    line = line.strip()
    if not line:
        return False
    if re.fullmatch(r"[\d\s\W]+", line):
        return True
    return None

def detect_line(line: str) -> bool:
    """
    Detects if a line contains only English words, keeping numerical data and symbols.

    Args:
        line: single line to check
    Returns:
        bool, True if line belongs to english, else False
    """ 
    if (result := is_valid_line(line)) is not None:
        return result
    try:
        return detect(line) == "en"
    except LangDetectException as e:
        print(f"[ERROR] Unable to detect language for line:\n'{line}'\nError: {e}\n")
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error for line:\n'{line}'\nError: {e}\n")
        return False

def process_chunk(chunk):
    """
    Process a chunk of the DataFrame for English lines
    """
    return chunk[chunk["text"].apply(detect_line)]

def main():
    ### Load data
    # Setup path to get data
    current_dir = os.getcwd()
    current_dir = Path(current_dir)
    data_dir = current_dir / "datasets" / "preprocessing" / "empty_lines_filtered.csv"
    # Load data into df
    all_data_df_filtered = pd.read_csv(data_dir)

    ### Get all valid lines (using multiprocessing)
    num_workers = os.cpu_count()
    chunk_size = num_workers
    chunks = np.array_split(all_data_df_filtered, chunk_size)

    with Pool(processes=chunk_size) as pool:
        results = pool.map(process_chunk, chunks)

    # Store into df
    english_data_df = pd.concat(results, ignore_index=True)
    # Save df to path
    output_dir = current_dir / "datasets" / "preprocessing"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "english_lang.csv"
    english_data_df.to_csv(output_file, encoding="utf-8")
    print(f"Preprocessed and saved to: {output_file}")

if __name__ == "__main__":
    main()