from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

import re

def clean_summary(text):
    text = re.sub(r'\b(julian zelizer):?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+[.,]', '.', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\.\s*:', '.', text)
    text = re.sub(r'\s*:\s*', ': ', text)
    text = re.sub(r':\s*\.', '.', text)
    text = re.sub(r'\bthan ever before\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*e\.g\.\s*$', '.', text)  # Remove dangling e.g.
    text = re.sub(r'-?\s*[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\.?', '', text)
    return text.strip()

def capitalize_first_letter(text):
    paragraphs = text.split('\n')
    capitalized_paragraphs = [p.strip().capitalize() for p in paragraphs]
    return '\n'.join(capitalized_paragraphs)

def remove_random_dates(text):
    # Remove patterns like day/time combos or dates
    patterns = [
        r'\b(thursday|monday|tuesday|wednesday|friday|saturday|sunday)\b\s*(night|morning|afternoon)?\s*(at)?\s*\d{1,2}(am|pm)?\s*(gmt)?',
        r'\b\d{1,2}(:\d{2})?\s*(am|pm)\b',
        r'\b(october|november|december|january|february|march|april|may|june|july|august|september)\s*\d{1,2}\b',
        r'\b\d{1,2}\s*(a\.m\.|p\.m\.|am|pm)\b',
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return text.strip()

def remove_incomplete_end(text):
    # Remove incomplete endings like dangling hyphens or partial times
    text = re.sub(r'[-–—]\s*p\.m\.$', '.', text)
    text = re.sub(r'\s*-\s*$', '', text)
    return text.strip()


model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name)


def summarize_paragraph(paragraph, compression_ratio=0.5):
    inputs = tokenizer(
        "summarize: " + paragraph,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding="longest"  # No excessive padding, just to longest seq in batch (here only one)
    )
    
    input_ids = inputs['input_ids'][0]
    input_length = (input_ids != tokenizer.pad_token_id).sum().item()  # real token count excluding padding

    max_summary_length = min(150, max(60, int(len(tokenizer.encode(paragraph)) * compression_ratio)))


# and then pass it here as:
    summary_ids = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_length=max_summary_length,
    min_length=40,
    num_beams=6,
    do_sample=False,
    no_repeat_ngram_size=2,
    repetition_penalty=2.0
)


    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    summary = remove_random_dates(summary)
    summary = remove_incomplete_end(summary)
    summary = clean_summary(summary)
    summary = capitalize_first_letter(summary)
    
    return summary


def summarize_text(paragraphs, compression_ratio=0.5):
    if isinstance(paragraphs, str):
        paragraphs = [p.strip() for p in paragraphs.split('\n') if p.strip()]
    
    summaries = [summarize_paragraph(p, compression_ratio) for p in paragraphs]
    return "\n\n".join(summaries)

if __name__ == "__main__":
    print("Enter paragraphs to summarize (separate each by a blank line). Press Ctrl+D (Linux/Mac) or Ctrl+Z (Windows) when done:\n")

    # Read multiline input from user until EOF
    user_input = []
    try:
        while True:
            line = input()
            user_input.append(line)
    except EOFError:
        pass

    # Join lines and split paragraphs by empty lines
    text = "\n".join(user_input)
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    print("\n🔹 Summarized Text 🔹\n")
    summarized_output = summarize_text(paragraphs, compression_ratio=0.5)
    print(summarized_output)