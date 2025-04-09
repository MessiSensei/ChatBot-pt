# downloading and cleaning dataset for our project ... Still needs some more cleaning but requires a lot of resources 
from datasets import load_dataset
import os

# Output folder
output_dir = "datasets_preprocessed"
os.makedirs(output_dir, exist_ok=True)

# 1. DailyDialog
print("\n Downloading DailyDialog...")
dailydialog = load_dataset("daily_dialog", trust_remote_code=True)

print(" Preprocessing DailyDialog...")
with open(os.path.join(output_dir, "dailydialog_input_response.txt"), "w", encoding="utf-8") as f:
    for split in ["train", "validation", "test"]:
        for dialog in dailydialog[split]:
            utterances = dialog["dialog"]
            for i in range(len(utterances) - 1):
                input_line = utterances[i].strip().replace("\n", " ")
                response_line = utterances[i + 1].strip().replace("\n", " ")
                f.write(f"{input_line}|{response_line}\n")

print(" Done: DailyDialog\n")

# 2. Cornell Movie Dialogs 
print(" Downloading Cornell Movie Dialogs...")
cornell = load_dataset("cornell-movie-dialog/cornell_movie_dialog")

print(" Preprocessing Cornell Movie Dialogs...")
with open(os.path.join(output_dir, "cornell_input_response.txt"), "w", encoding="utf-8") as f:
    for dialog in cornell["train"]:
        lines = dialog["lines"]
        for i in range(len(lines) - 1):
            input_line = lines[i]["text"].strip().replace("\n", " ")
            response_line = lines[i + 1]["text"].strip().replace("\n", " ")
            f.write(f"{input_line}|{response_line}\n")

print(" Done: Cornell Movie Dialogs\n")

#  3. PersonaChat 
print(" Downloading PersonaChat...")
personachat = load_dataset("AlekseyKorshuk/persona-chat")

print(" Preprocessing PersonaChat...")
with open(os.path.join(output_dir, "personachat_input_response.txt"), "w", encoding="utf-8") as f:
    for split in ["train", "validation", "test"]:
        for dialog in personachat[split]:
            utterances = dialog["history"]
            if len(utterances) % 2 != 0:
                utterances = utterances[:-1]
            for i in range(0, len(utterances) - 1, 2):
                input_line = utterances[i].strip().replace("\n", " ")
                response_line = utterances[i + 1].strip().replace("\n", " ")
                f.write(f"{input_line}|{response_line}\n")

print(" Done: PersonaChat\n")

print(f" All datasets processed and saved in: {output_dir}/")
