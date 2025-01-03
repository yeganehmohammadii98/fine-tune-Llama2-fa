from datasets import Dataset
import torch
import logging
import os
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# Setup logging function remains the same
def setup_logging():
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

# prepare_dataset function remains the same
def prepare_dataset():
    logging.info("Starting to load dataset...")
    try:
        with open('combined_texts.txt', 'r', encoding='utf-8') as f:
            texts = f.read().split('\n\n')
            texts = [t.strip() for t in texts if t.strip()]
        
        dataset = Dataset.from_dict({"text": texts})
        logging.info(f"Dataset loaded successfully with {len(texts)} examples")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {str(e)}")
        raise

# Modified setup_model_and_tokenizer function
def setup_model_and_tokenizer(model_name="facebook/opt-1.3b"):
    logging.info(f"Setting up model and tokenizer for {model_name}")
    try:
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        # Load tokenizer
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model
        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        model = prepare_model_for_kbit_training(model)
        logging.info("Model and tokenizer setup completed")
        return model, tokenizer
    except Exception as e:
        logging.error(f"Error in model/tokenizer setup: {str(e)}")
        raise

# Modified setup_lora function with OPT-specific target modules
def setup_lora(model):
    logging.info("Setting up LoRA...")
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            # Modified target modules for OPT model
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        logging.info("LoRA setup completed")
        return model
    except Exception as e:
        logging.error(f"Error in LoRA setup: {str(e)}")
        raise

# tokenize_dataset function remains the same
def tokenize_dataset(dataset, tokenizer):
    logging.info("Starting dataset tokenization...")
    try:
        def tokenize(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        logging.info("Dataset tokenization completed")
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Error in dataset tokenization: {str(e)}")
        raise

# Modified train function with adjusted output directories
def train():
    logging.info("Starting training process...")
    try:
        # Load and prepare dataset
        dataset = prepare_dataset()
        
        # Setup model and tokenizer
        model, tokenizer = setup_model_and_tokenizer()
        
        # Setup LoRA
        model = setup_lora(model)
        
        # Tokenize dataset
        tokenized_dataset = tokenize_dataset(dataset, tokenizer)
        
        # Split dataset
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
        logging.info(f"Dataset split completed. Train size: {len(split_dataset['train'])}, Test size: {len(split_dataset['test'])}")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./opt-finetuned",  # Changed directory name
            num_train_epochs=3,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=8,
            evaluation_strategy="steps",
            eval_steps=100,
            save_steps=100,
            warmup_steps=100,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            optim="paged_adamw_8bit",
            save_total_limit=3,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        logging.info("Initializing Trainer...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            data_collator=data_collator,
        )
        
        # Start training
        logging.info("Starting training...")
        trainer.train()
        
        # Save the model
        logging.info("Saving final model...")
        trainer.save_model("./final-opt-model")  # Changed save directory
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    setup_logging()
    train()
