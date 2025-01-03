import subprocess
import sys
import pkg_resources
import os

def install_required_packages():
    """Install required packages if they're not already installed"""
    required_packages = {
        'transformers': '4.31.0',
        'torch': '2.0.0',
        'datasets': '2.14.0',
        'peft': '0.4.0',
        'bitsandbytes': '0.41.0',
        'accelerate': '0.21.0',
        'sentencepiece': '0.1.99'
    }
    
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    
    packages_to_install = []
    for package, version in required_packages.items():
        if package not in installed_packages:
            packages_to_install.append(f"{package}>={version}")
        elif pkg_resources.parse_version(installed_packages[package]) < pkg_resources.parse_version(version):
            packages_to_install.append(f"{package}>={version}")
    
    if packages_to_install:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages_to_install)
        print("Required packages installed successfully!")
    else:
        print("All required packages are already installed!")

# Install required packages
install_required_packages()

# Now import the required packages
import torch
import logging
from datetime import datetime
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from accelerate import Accelerator

# Set your HF token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Rest of your code continues here...

def setup_logging():
    """Initialize logging configuration"""
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

def verify_setup():
    """Verify CUDA setup"""
    logging.info("Verifying CUDA setup...")
    logging.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logging.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logging.info(f"Device name: {torch.cuda.get_device_name()}")
        logging.info(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

def prepare_dataset():
    """Load and prepare the dataset"""
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

def setup_model_and_tokenizer(model_name="meta-llama/Llama-2-7b-hf"):
    """Initialize model and tokenizer with optimizations"""
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
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"],
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=os.environ["HUGGING_FACE_HUB_TOKEN"],
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

def setup_lora(model):
    """Configure LoRA adaptors"""
    logging.info("Setting up LoRA...")
    try:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
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

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset"""
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

def train():
    """Main training function"""
    try:
        # Initialize logging
        setup_logging()
        
        # Verify CUDA setup
        verify_setup()
        
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
            output_dir="./llama2-7b-finetuned",
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
        trainer.save_model("./final-llama2-7b-model")
        logging.info("Training completed successfully!")
        
    except Exception as e:
        logging.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    train()
