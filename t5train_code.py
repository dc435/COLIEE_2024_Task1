# COLIEEE 2024

# =========================================
# T5TRAIN_CODE
# Code to fine-tune t5-base model to extract proposition from TARGETCASE paragraph
# =========================================



import torch, random, logging, sys
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    stream=sys.stdout)

# Helper to set seed for reproducability:
def set_seed(seed):

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Method to build t5 trainer for fine-tuning
def get_trainer():

    # Set a fixed seed for reproducibility
    seed = 42
    logging.info('Set random seed to %d' % seed)
    set_seed(42)

    # Base model details at: https://huggingface.co/t5-base

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logging.info('Building trainer on device: %s' % device)

    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    data = load_dataset("csv", data_files='./files/t5_train_data.csv')
    max_input_length = 512
    max_target_length = 128

    def preprocess_function(datum):
        inputs = datum['input']
        model_inputs = tokenizer(inputs, max_length=max_input_length)

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(datum['output'], max_length=max_target_length)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_datasets = data.map(preprocess_function, batched=True)

    batch_size = 8
    args = Seq2SeqTrainingArguments(
        "model-training",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=20,
        predict_with_generate=True,
        save_steps=10000000
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    return trainer

# Method to t5 train, and save model weights:
def train_save_model(trainer):

    logging.info('Training model')

    trainer.train()
    path_training = "./models/trained_t5"
    trainer.save_model(path_training)

    logging.info('Saved model to %s' % path_training)
