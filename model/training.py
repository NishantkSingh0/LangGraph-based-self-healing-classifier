from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

def prepare_model():
    model=AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    config=LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=32, lora_dropout=0.1,
        bias="none"
    )
    model=get_peft_model(model, config)
    return model

def train_model(model, tokenized_dataset):
    dataset=Dataset.from_dict(tokenized_dataset)
    dataset=dataset.train_test_split(test_size=0.1)

    training_args=TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs"
    )

    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer.train()
    return model
