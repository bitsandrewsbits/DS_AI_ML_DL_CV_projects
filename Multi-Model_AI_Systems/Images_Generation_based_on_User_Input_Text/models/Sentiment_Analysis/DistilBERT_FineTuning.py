# DistilBERT model fine-tuning
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, TFAutoModelForSequenceClassification
from transformers import create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
import tensorflow as tf

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
        "distilbert/distilbert-base-uncased", num_labels = 2,
        id2label = ID_to_label, label2id = label_to_ID
    )

    batch_size = 10
    epochs_amount = 40
    init_learing_rate = 1e-5
    warmup_steps = 0
    batches_per_epoch = len(tokenized_datasets["train"]) // batch_size
    total_train_batch_steps = int(batches_per_epoch * epochs_amount)
    optimizer, schedule = get_optimizer_and_schedule(
        init_learing_rate, warmup_steps, total_train_batch_steps
    )

    train_val_test_tf_datasets = get_converted_all_datasets(
        model, tokenized_datasets, batch_size, data_collator
    )
    accuracy_callback = KerasMetricCallback(
        metric_fn = compute_accuracy_metric,
        eval_dataset = train_val_test_tf_datasets["validation"]
    )
    model.compile(optimizer = tf.keras.optimizers.Adam(3e-5))

def load_train_val_test_datasets(datasets_parent_dir_path: str) -> DatasetDict:
    loaded_datasets = DatasetDict.load_from_disk(
        dataset_dict_path = datasets_parent_dir_path
    )
    return loaded_datasets

def compute_accuracy_metric(evaluated_prediction):
    predictions, labels = evaluated_prediction
    predictions = np.argmax(predictions, axis = 1)
    return accuracy.compute(predictions = predictions, references = labels)

def get_optimizer_and_schedule(learning_rate, warmup_steps, train_steps):
    return create_optimizer(
        init_lr = learning_rate, num_warmup_steps = warmup_steps,
        num_train_steps = train_steps
    )

def get_converted_all_datasets(
model, datasets: DatasetDict, batch_size, collate_func) -> tf.data.Dataset:
    tf_datasets = {}
    for dataset_type in datasets:
        tf_datasets[dataset_type] = get_converted_tf_dataset(
            model, datasets[dataset_type], batch_size, collate_func
        )
    return tf_datasets

def get_converted_tf_dataset(
model, dataset: Dataset, batch_size, collate_func) -> tf.data.Dataset:
    return model.prepare_tf_dataset(
        dataset,
        shuffle = True,
        batch_size = batch_size,
        collate_fn = collate_func
    )

if __name__ == '__main__':
    main()
