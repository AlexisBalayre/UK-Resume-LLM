"""
This script demonstrates the process of fine-tuning a large language model using the Hugging Face Transformers library, BitsAndBytes for memory optimization, and PEFT (Parameter Efficient Fine Tuning) techniques for efficient training. It involves loading and preparing a model and tokenizer, configuring training parameters, performing the training process, and saving the fine-tuned model. It specifically focuses on causal language modeling with a custom dataset.
"""

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from datasets import load_dataset
import torch
from trl import SFTTrainer

# Initial configuration for model and tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "AlexisBalayre/who-am-I_mistral_7b"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.padding_side = "right"

# Configuration for BitsAndBytes to reduce memory footprint
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)

# Load and prepare the model for training with memory optimization and automatic device mapping
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",  # Automatically distribute the model across available GPUs
)
model.config.use_cache = False  # Important for certain types of training
model = prepare_model_for_kbit_training(model)

# Configure the pad token in the model to match the tokenizer
model.config.pad_token_id = tokenizer.pad_token_id

# PEFT-specific configuration for the model to introduce parameter-efficient techniques
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
)
model = get_peft_model(model, peft_config)

# Load and tokenize datasets for training and validation
dataset = load_dataset(
    "json", data_files={"train": "data/train.jsonl", "validation": "data/valid.jsonl"}
)


def tokenize_function(examples):
    """
    Tokenizes text examples using the provided tokenizer. Ensures consistent padding and truncation.

    Parameters:
    - examples (dict): A batch of text examples.

    Returns:
    - dict: The tokenized representations of the inputs.
    """
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configuration des arguments d'entraînement
training_arguments = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,  # Reduced for better GPU memory management
    per_device_eval_batch_size=8,  # Consistent with training batch size for consistency
    gradient_accumulation_steps=4,  # Increased to allow for effectively larger batch sizes
    optim="adamw_torch",  # Switch to a more standard optimizer for potentially better performance
    save_steps=1000,  # Less frequent saves to focus on training
    logging_steps=100,  # More frequent logging to closely monitor progress
    learning_rate=2e-5,  # Lower learning rate for fine-tuning on a smaller dataset
    fp16=True,  # Enable fp16 to speed up training, assuming compatible hardware is available
    gradient_checkpointing=True,  # Enable to save memory at the cost of slower training
    evaluation_strategy="epoch",  # Evaluate at the end of each epoch for more stable metrics
    save_strategy="epoch",  # Save at the end of each epoch for more stable checkpoints
    max_grad_norm=1.0,  # Increased to allow larger updates (can help with convergence)
    num_train_epochs=3.0,  # More epochs for more thorough learning
    weight_decay=0.01,  # Increased weight decay for regularization
    warmup_steps=100,  # Increased warmup steps for a smoother learning rate transition
    lr_scheduler_type="cosine",  # Use cosine learning rate scheduler for smoother decays
    load_best_model_at_end=True,  # Load the best model at the end of training
    metric_for_best_model="eval_loss",  # Use eval loss to determine the best model
    greater_is_better=False,  # Since we're using loss to determine the best model, lower is better
)

# Initialize the trainer with model, datasets, PEFT configuration, and training arguments
trainer = SFTTrainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# Start the training process
trainer.train()

# Save the fine-tuned model to a specified path
trainer.model.save_pretrained(new_model)

# Clear the memory footprint after training is complete
del model, trainer
torch.cuda.empty_cache()

# Example for reloading the base model with memory optimization settings and merging PEFT parameters
base_model_reload = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
)
model = PeftModel.from_pretrained(base_model_reload, new_model)
model = model.merge_and_unload()

# Save the merged model and reload tokenizer to ensure consistency
model_dir = "./merged_model"
model.save_pretrained(model_dir, safe_serialization=True)
tokenizer.save_pretrained(model_dir)
