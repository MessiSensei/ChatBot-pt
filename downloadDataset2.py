# downloading and cleaning dataset for our project ... Still needs some more cleaning but requires a lot of resources 
 
from datasets import load_dataset
import os
import re
import json

# Output folder
output_dir = "datasets2_5000"
os.makedirs(output_dir, exist_ok=True)

# Filter thresholds
MIN_LEN = 5
MAX_LEN = 5000

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s([?.!,])", r"\1", text)
    return text.strip()

def is_valid_pair(input_text, response_text):
    return all([
        MIN_LEN <= len(input_text) <= MAX_LEN,
        MIN_LEN <= len(response_text) <= MAX_LEN
    ])

txt_combined = open(os.path.join(output_dir, "all_input_response.txt"), "w", encoding="utf-8")
jsonl_combined = open(os.path.join(output_dir, "all_input_response.jsonl"), "w", encoding="utf-8")

def write_pair(txt_file, jsonl_file, input_text, response_text):
    input_clean = clean_text(input_text)
    response_clean = clean_text(response_text)
    if is_valid_pair(input_clean, response_clean):
        txt_file.write(f"{input_clean}|{response_clean}\n")
        json.dump({"input": input_clean, "response": response_clean}, jsonl_file)
        jsonl_file.write("\n")

#  DailyDialog
print("\n Downloading DailyDialog...")
dailydialog = load_dataset("daily_dialog", trust_remote_code=True)

print(" Preprocessing DailyDialog...")
with open(os.path.join(output_dir, "dailydialog_input_response.txt"), "w", encoding="utf-8") as ftxt,      open(os.path.join(output_dir, "dailydialog_input_response.jsonl"), "w", encoding="utf-8") as fjsonl:
    for split in ["train", "validation", "test"]:
        for dialog in dailydialog[split]:
            utterances = dialog["dialog"]
            for i in range(len(utterances) - 1):
                write_pair(ftxt, fjsonl, utterances[i], utterances[i + 1])
                write_pair(txt_combined, jsonl_combined, utterances[i], utterances[i + 1])
print(" Done: DailyDialog\n")

#  Cornell Movie Dialogs 
print(" Downloading Cornell Movie Dialogs...")
cornell = load_dataset("cornell-movie-dialog/cornell_movie_dialog")

print(" Preprocessing Cornell Movie Dialogs...")
with open(os.path.join(output_dir, "cornell_input_response.txt"), "w", encoding="utf-8") as ftxt, \
     open(os.path.join(output_dir, "cornell_input_response.jsonl"), "w", encoding="utf-8") as fjsonl:

    for row in cornell["train"]:
        utterance = row.get("utterance", {})
        texts = utterance.get("text", [])
        if isinstance(texts, list) and len(texts) > 1:
            for i in range(len(texts) - 1):
                write_pair(ftxt, fjsonl, texts[i], texts[i + 1])
                write_pair(txt_combined, jsonl_combined, texts[i], texts[i + 1])
print(" Done: Cornell Movie Dialogs")


# ========== PersonaChat ==========
print(" Downloading PersonaChat...")
personachat = load_dataset("AlekseyKorshuk/persona-chat")

print(" Preprocessing PersonaChat...")
with open(os.path.join(output_dir, "personachat_input_response.txt"), "w", encoding="utf-8") as ftxt,      open(os.path.join(output_dir, "personachat_input_response.jsonl"), "w", encoding="utf-8") as fjsonl:
    for split in personachat:
        for dialog in personachat[split]:
            for turn in dialog.get("utterances", []):
                history = turn.get("history", [])
                for i in range(len(history) - 1):
                    write_pair(ftxt, fjsonl, history[i], history[i + 1])
                    write_pair(txt_combined, jsonl_combined, history[i], history[i + 1])
print(" Done: PersonaChat\n")

txt_combined.close()
jsonl_combined.close()
print(f" All datasets processed and saved to '{output_dir}' with length filtering and both .txt/.jsonl formats.")