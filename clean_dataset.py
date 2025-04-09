import os
from textblob import TextBlob

INPUT_FILE = "data/dataset.txt"
OUTPUT_FILE = "data/dataset_cleaned.txt"

def is_valid_pair(line):
    return "|" in line and len(line.split("|")) == 2

def clean_text(text):
    blob = TextBlob(text.strip())
    return str(blob.correct()).strip()

def clean_dataset(input_path, output_path):
    seen = set()
    cleaned_lines = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if not is_valid_pair(line):
                continue
            input_part, response_part = line.strip().split("|", 1)
            if len(input_part.split()) < 2 or len(response_part.split()) < 2:
                continue

            input_cleaned = clean_text(input_part)
            response_cleaned = clean_text(response_part)

            if len(input_cleaned.split()) < 2 or len(response_cleaned.split()) < 2:
                continue

            final_line = f"{input_cleaned}|{response_cleaned}"
            if final_line not in seen:
                seen.add(final_line)
                cleaned_lines.append(final_line)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for line in cleaned_lines:
            out_f.write(line + "\n")

    print(f"Cleaned dataset saved to: {output_path}")
    print(f"Total cleaned pairs: {len(cleaned_lines)}")

if __name__ == "__main__":
    clean_dataset(INPUT_FILE, OUTPUT_FILE)
