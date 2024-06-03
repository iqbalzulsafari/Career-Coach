import pandas as pd
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("waelChafei/interviewtest")

# Get the train dataset
train_dataset = dataset["train"]

# Convert the train dataset to a Pandas DataFrame
df = pd.DataFrame(train_dataset)

# Save the DataFrame as a CSV file
df.to_csv("interview_dataset.csv", index=False)