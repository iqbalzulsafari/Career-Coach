import spacy
import random
from spacy.training.example import Example

# Function to download the model if it's not already present
def download_model(model_name):
    try:
        spacy.load(model_name)
        print(f"Model '{model_name}' is already installed.")
    except OSError:
        print(f"Model '{model_name}' not found. Downloading...")
        spacy.cli.download(model_name)

# Download the en_core_web_sm model
download_model("en_core_web_sm")

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")
ner = nlp.get_pipe("ner")

# Sample training data
TRAIN_DATA = [
    ("Uber blew through $1 million a week", {"entities": [(0, 4, "ORG")]}),
    ("Google rebrands its business apps", {"entities": [(0, 6, "ORG")]}),
    # Add more examples here
]

# Add new labels
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipeline components
pipe_exceptions = ["ner"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]

# Train the NER model
with nlp.disable_pipes(*unaffected_pipes):
    optimizer = nlp.resume_training()
    for itn in range(30):
        random.shuffle(TRAIN_DATA)
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], drop=0.35, losses=losses)
        print(f"Losses at iteration {itn}: {losses}")

# Save the trained model
nlp.to_disk("ner_model")

# Load and test the model
nlp2 = spacy.load("ner_model")
doc = nlp2("Google rebrands its business apps")
for ent in doc.ents:
    print(ent.text, ent.label_)
