# DistilBERT model fine-tuning
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TFAutoModelForSequenceClassification
from transformers import create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

datasets_parent_dir = "data_collection_and_preprocessing"
accuracy = evaluate.load("accuracy")

ID_to_label = {0: "negative", 1: "positive", 2: "neutral"}
label_to_ID = {"negative": 0, "positive": 1, "neutral": 2}

def main():
    train_val_test_datasets = load_train_val_test_datasets(datasets_parent_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenized_datasets = train_val_test_datasets.map(
        lambda dataset: tokenizer(dataset["text"], truncation = True),
        batched = True
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["text"])

    data_collator = DataCollatorWithPadding(tokenizer = tokenizer)

    distilbert_model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels = 3,
        id2label = ID_to_label, label2id = label_to_ID,
        device_map = "auto"
    )

    training_args = TrainingArguments(
        output_dir = "fine_tuned_DistilBERT",
        learning_rate = 2e-5,
        per_device_train_batch_size = 60,
        per_device_eval_batch_size = 60,
        num_train_epochs = 2,
        weight_decay = 0.01,
        eval_strategy = "steps",
        logging_strategy = 'steps',
        save_strategy = "steps",
        logging_steps = 1,
        load_best_model_at_end = True,
        report_to = "tensorboard"
    )

    trainer = Trainer(
        model = distilbert_model,
        args = training_args,
        train_dataset = tokenized_datasets["train"],
        eval_dataset = tokenized_datasets["validation"],
        processing_class = tokenizer,
        data_collator = data_collator,
        compute_metrics = compute_accuracy_metric,
    )
    trainer.train()

    # model_prediction = model.evaluate(
    #     train_val_test_datasets["test"]
    # )
    # print(classification_report(
    #     y_true = train_val_test_datasets["test"],
    #     y_pred = model_prediction,
    #     labels = [0, 1, 2],
    #     target_names = ["negative", "positive", "neutral"]
    # ))

def load_train_val_test_datasets(datasets_parent_dir_path: str) -> DatasetDict:
    loaded_datasets = DatasetDict.load_from_disk(
        dataset_dict_path = datasets_parent_dir_path
    )
    return loaded_datasets

def compute_accuracy_metric(evaluated_prediction):
    predictions, labels = evaluated_prediction
    predictions = np.argmax(predictions, axis = 1)
    return accuracy.compute(predictions = predictions, references = labels)

if __name__ == '__main__':
    main()
