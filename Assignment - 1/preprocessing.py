from datasets import load_dataset
from transformers import BertTokenizer

def load_and_preprocess_data(dataset_name='conll2003'):
    # Load dataset
    dataset = load_dataset(dataset_name)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
        labels = []
        for i, label in enumerate(examples['ner_tags']):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)  # Special token
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)  # Word continuation
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply function to each split
    dataset = dataset.map(tokenize_and_align_labels, batched=True)
    return dataset

dataset = load_and_preprocess_data()
