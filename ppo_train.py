pip install -qU transformers==4.45.2 accelerate bitsandbytes peft trl==0.11 datasets tqdm

from transformers import AutoTokenizer
from trl import AutoModelForSeq2SeqLMWithValueHead
import torch

import pandas as pd
from datasets import Dataset

# Load the supcom dataset from Google Colab path
df = pd.read_csv("/content/supcom_dataset.csv")

# Convert the pandas DataFrame into Hugging Face dataset
dataset = Dataset.from_pandas(df)

def map_initial_prompts(sample):
  return {"Query" : sample["chosen"].split("chunk:")[0]}

dataset = dataset.map(map_initial_prompts)

from accelerate import Accelerator
current_device = Accelerator().local_process_index

from huggingface_hub import notebook_login

notebook_login()

from transformers import AutoModelForSequenceClassification
import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer

reward_model_id = "distilroberta-base"

reward_model = AutoModelForSequenceClassification.from_pretrained(
    reward_model_id,
    num_labels=1,
    device_map={"" : current_device},
)
reward_model_tokenizer = AutoTokenizer.from_pretrained(reward_model_id)

# classic postprocessing for padding/eos_token issues
if reward_model_tokenizer.pad_token is None:
    reward_model_tokenizer.pad_token = reward_model_tokenizer.eos_token
    reward_model_id.config.pad_token_id = reward_model_id.config.eos_token_id

def formatting_function(sample):
  kwargs = {
      "padding" : "max_length",
      "truncation" : True,
      "max_length" : 512,
      "return_tensors" : "pt"}

  chosen_tokens = reward_model_tokenizer.encode_plus(sample["chosen"], **kwargs)
  rejected_tokens = reward_model_tokenizer.encode_plus(sample["rejected"], **kwargs)

  return {
        "input_ids_chosen": chosen_tokens["input_ids"][0], "attention_mask_chosen": chosen_tokens["attention_mask"][0],
        "input_ids_rejected": rejected_tokens["input_ids"][0], "attention_mask_rejected": rejected_tokens["attention_mask"][0]
    }

formatted_dataset =dataset .map(formatting_function)

from trl import RewardConfig

reward_config = RewardConfig(
    output_dir="/content/reward_model",
    per_device_train_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=50,
    logging_steps=1,
    max_steps = 500,
    report_to=None,
    center_rewards_coefficient=0.01,
)

from trl import RewardTrainer

trainer = RewardTrainer(
    model=reward_model,
    args=reward_config,
    tokenizer=reward_model_tokenizer,
    train_dataset=formatted_dataset,
    eval_dataset=formatted_dataset.select(range(3)),
)

# Train the reward model
trainer.train()

trainer.save_model()

reward_model = AutoModelForSequenceClassification.from_pretrained(
    "/content/reward_model",
    device_map={"" : current_device},
)

# Ensure reward_model is on the GPU
reward_model = reward_model.to(device)

from transformers import AutoTokenizer
from trl import AutoModelForSeq2SeqLMWithValueHead
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("t5-small")  # You can use any compatible model
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained("t5-small")

# Example input: a query and a list of candidate documents
query = "What are the benefits of using renewable energy?"
documents = [
    "Renewable energy sources are sustainable and reduce carbon emissions.",
    "Using fossil fuels contributes to global warming.",
    "Renewable energy can create jobs in new industries.",
    "Solar panels are a common form of renewable energy.",
    "Investing in renewable energy is essential for a sustainable future."
]

# Prepare inputs for the model
input_texts = [f"query: {query} passage: {doc}" for doc in documents]
inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")

# Prepare decoder input (using decoder start token)
decoder_input_ids = torch.full(
    (inputs['input_ids'].shape[0], 1),
    model.config.decoder_start_token_id,
    dtype=torch.long
)

# Forward pass through the model with return_dict=True
with torch.no_grad():
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        decoder_input_ids=decoder_input_ids,
        return_dict=True  # Ensure the output is a structured object
    )

scores = outputs[2].squeeze()  # Remove any unnecessary dimensions

# Get the ranking based on scores (higher score = more relevant)
ranking_indices = torch.argsort(scores, descending=True)

# Display ranked documents
print("Ranked Documents:")
for idx in ranking_indices.tolist():
    print(documents[idx])

from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
from peft import LoraConfig

# Define PPO configuration with batch size of 128
ppo_config = PPOConfig(
    model_name="t5-small",
    learning_rate=1e-5,
    batch_size=128,  # Required batch size for PPO
    ppo_epochs=4,

)

# Initialize PPO trainer with the model and tokenizer
ppo_trainer = PPOTrainer(model=model, tokenizer=tokenizer, config=ppo_config)

# Set gradient accumulation steps to accumulate 128 examples (since we only have 10 rows)
accumulation_steps = 13  # 128 / 10 ≈ 13 steps of gradient accumulation

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad()

# Function to calculate the reward score for each query-document pair
def get_reward_score(query, doc):
    input_text = f"query: {query} document: {doc}"

    # Tokenize the input text for the reward model
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

    # Move inputs to the same device as the model
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get the logits (relevance score) from the reward model
    with torch.no_grad():
        logits = reward_model(**inputs).logits

    return logits.item()  # Return the score

# Function to calculate the reranker reward (using the reward model)
def calculate_reranker_reward(query, reranked_docs):
    reward_scores = [get_reward_score(query, doc) for doc in reranked_docs]
    doc_score_pairs = list(zip(reranked_docs, reward_scores))

    # Sort documents based on scores (higher score = better rank)
    sorted_docs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    sorted_order = [doc for doc, _ in sorted_docs]

    # Return reward if the order matches the expected order (sorted order)
    if reranked_docs == sorted_order:
        return 1  # Reward if the order is correct
    else:
        return -1  # Penalize if the order is wrong

device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0

# Define your input data
input_texts = [
    {
        "query": "What are the benefits of using renewable energy?",
        "documents": [
            "Renewable energy sources are sustainable and reduce carbon emissions.",
            "Using fossil fuels contributes to global warming.",
            "Renewable energy can create jobs in new industries.",
            "Solar panels are a common form of renewable energy.",
            "Investing in renewable energy is essential for a sustainable future."
        ]
    },
    {
        "query": "How does solar power contribute to sustainability?",
        "documents": [
            "Solar power reduces dependency on fossil fuels and lowers greenhouse gas emissions.",
            "Solar energy is renewable and abundant.",
            "Solar power systems can be installed on homes or businesses, making them decentralized.",
            "Solar power generation has low environmental impact compared to fossil fuels.",
            "Investing in solar power leads to long-term savings on energy bills."
        ]
    },
    {
        "query": "What are the economic impacts of wind energy?",
        "documents": [
            "Wind energy creates jobs in manufacturing, installation, and maintenance.",
            "Wind power can help reduce energy costs for consumers.",
            "The wind energy industry has a significant economic impact in rural areas.",
            "Investing in wind energy reduces dependency on imported fossil fuels.",
            "Wind energy can help meet renewable energy goals and reduce carbon emissions."
        ]
    }
]

from torch.utils.data import DataLoader

# Move models to device
model.to(device)
ref_model.to(device)
reward_model.to(device)

# PPO Configuration
config = PPOConfig(
    model_name="t5-small",
    learning_rate=1.41e-5,
    ppo_epochs=4,
    batch_size=2,
    mini_batch_size=1,
)

# Define dataset
dataset = [
    {
        "query": "What are the benefits of using renewable energy?",
        "documents": [
            "Renewable energy sources are sustainable and reduce carbon emissions.",
            "Using fossil fuels contributes to global warming.",
            "Renewable energy can create jobs in new industries.",
            "Solar panels are a common form of renewable energy.",
            "Investing in renewable energy is essential for a sustainable future."
        ]
    },
    {
        "query": "How does solar power contribute to sustainability?",
        "documents": [
            "Solar power reduces dependency on fossil fuels and lowers greenhouse gas emissions.",
            "Solar energy is renewable and abundant.",
            "Solar power systems can be installed on homes or businesses, making them decentralized.",
            "Solar power generation has low environmental impact compared to fossil fuels.",
            "Investing in solar power leads to long-term savings on energy bills."
        ]
    }
]

# DataLoader
def collate_fn(batch):
    return batch

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)

# Initialize PPO Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,

    tokenizer=tokenizer
)

# PPO Training Loop
max_ppo_steps = 10

for step, batch in enumerate(dataloader):
    if step >= max_ppo_steps:
        break

    batch = batch[0]
    query = batch["query"]
    documents = batch["documents"]

    # Prepare inputs for the model
    input_texts = [f"query: {query} passage: {doc}" for doc in documents]
    inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").to(device)

    # Generate reranked scores
    decoder_input_ids = torch.full(
        (inputs['input_ids'].shape[0], 1),
        model.config.decoder_start_token_id,
        dtype=torch.long
    ).to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
    scores = outputs.logits.squeeze()  # Assuming logits correspond to relevance scores
    ranking_indices = torch.argsort(scores, descending=True)

    # Generate reranked documents based on model output
    reranked_docs = [documents[idx] for idx in ranking_indices.tolist()]

    # Calculate reward
    reward = calculate_reranker_reward(query, reranked_docs)
    reward_tensor = torch.tensor([reward], dtype=torch.float).to(device)

    # Prepare tensors for PPO step
    prompt_tensors = inputs["input_ids"]
    response_tensors = tokenizer(reranked_docs, padding=True, truncation=True, return_tensors="pt")["input_ids"].to(device)

    # Perform PPO step
    stats = ppo_trainer.step(prompt_tensors, response_tensors, reward_tensor)

    # Log stats
    print(f"Step {step + 1}:")
    print(f"Reward: {reward}")
    print(f"Mean Reward: {stats['ppo/returns/mean']}")
    print(f"KL Divergence: {stats['objective/kl']}")
    print("-" * 50)