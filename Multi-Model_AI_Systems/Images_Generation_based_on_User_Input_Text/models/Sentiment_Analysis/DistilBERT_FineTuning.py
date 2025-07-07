# DistilBERT model fine-tuning
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, TFAutoModelForSequenceClassification
from datasets import DatasetDict
import evaluate
import numpy as np

datasets_parent_dir = "data_collection_and_preprocessing"
accuracy = evaluate.load("accuracy")

ID_to_label = {0: "negative", 1: "positive"}
label_to_ID = {"negative": 0, "positive": 1}

def main():
    train_val_test_datasets = load_train_val_test_datasets(datasets_parent_dir)
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    tokenized_datasets = train_val_test_datasets.map(
        lambda dataset: tokenizer(dataset["text"], truncation = True),
        batched = True
    )

    data_collator = DataCollatorWithPadding(
        tokenizer = tokenizer, return_tensors = "tf"
    )

    training_args = TrainingArguments(
        output_dir = "fine_tuned_DistilBERT",
        eval_strategy = "epoch"
    )

    model = TFAutoModelForSequenceClassification.from_pretrained(
        "distilbert/distilbert-base-uncased", num_label = 2,
        id2label = ID_to_label, label2id = label_to_ID
    )

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
