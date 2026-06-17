import json
import os
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, VALID_LABELS, DATA_PATH, TRAIN_FILE, LABELS_FILE

_client = Groq(api_key=GROQ_API_KEY)


def load_labeled_examples() -> list[dict]:
    """
    Load the training episodes and merge them with the student's labels.

    Returns a list of dicts, each with:
      - "id"          : episode ID
      - "title"       : episode title
      - "podcast"     : podcast name
      - "description" : episode description
      - "label"       : the label from my_labels.json (may be None if not yet annotated)

    Only returns episodes where the label is a valid, non-null string.
    Episodes with null labels are silently skipped.
    """
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    labels_path = os.path.join(DATA_PATH, LABELS_FILE)

    with open(train_path, encoding="utf-8") as f:
        episodes = {ep["id"]: ep for ep in json.load(f)}

    with open(labels_path, encoding="utf-8") as f:
        labels = {entry["id"]: entry["label"] for entry in json.load(f)}

    labeled = []
    for ep_id, ep in episodes.items():
        label = labels.get(ep_id)
        if label in VALID_LABELS:
            labeled.append({**ep, "label": label})

    return labeled


def build_few_shot_prompt(labeled_examples: list[dict], description: str) -> str:
    """
    Build a few-shot classification prompt using labeled training examples.
    """
    label_definitions = """You are a podcast format classifier. Classify podcast episode descriptions into exactly one of these four format labels:

- interview: A host speaks with one or more guests, structured around questions and responses. Clear host-guest dynamic.
- solo: One host speaks alone — personal essay, opinion, reflection, tutorial. No guests.
- panel: Three or more speakers discuss a topic as rough equals. No clear host-guest dynamic.
- narrative: A story told using reporting, multiple sources, audio clips, or documentary-style production. Has a story arc.

Key distinctions:
- A personal story told from memory by one host = solo (not narrative)
- A story assembled from external documents, archives, or interviews = narrative
- Two people where one clearly leads = interview (not panel)
- Three or more people with roughly equal standing = panel (not interview)
"""

    examples_block = "Here are labeled examples:\n\n"
    for ex in labeled_examples:
        examples_block += f"Description: {ex['description']}\nLabel: {ex['label']}\n\n"

    instruction = f"""Now classify this new episode description.

Description: {description}

Respond in exactly this format:
Label: <one of: interview, solo, panel, narrative>
Reasoning: <one sentence explaining why>"""

    return label_definitions + examples_block + instruction


def classify_episode(description: str, labeled_examples: list[dict]) -> dict:
    """
    Classify a single podcast episode description using the few-shot LLM classifier.
    """
    try:
        prompt = build_few_shot_prompt(labeled_examples, description)

        response = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        response_text = response.choices[0].message.content.strip()

        # Parse "Label: X" and "Reasoning: Y" from response
        label = "unknown"
        reasoning = response_text

        for line in response_text.splitlines():
            line_stripped = line.strip()
            if line_stripped.lower().startswith("label:"):
                raw_label = line_stripped[len("label:"):].strip().lower()
                # Remove any markdown formatting like ** or *
                raw_label = raw_label.strip("*_ ")
                if raw_label in VALID_LABELS:
                    label = raw_label
            elif line_stripped.lower().startswith("reasoning:"):
                reasoning = line_stripped[len("reasoning:"):].strip()

        return {"label": label, "reasoning": reasoning}

    except Exception as e:
        return {"label": "unknown", "reasoning": f"Classification failed: {e}"}
