import torch
# from transformers import BertTokenizerFast
from datasets import load_dataset

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS device not found.")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class NERDataset():

    def __init__(self, dataset, tokenizer):
        self.dataset = load_dataset(dataset)
        # self.dataset = load_dataset("conll2003")
        self.tokenizer = tokenizer
        # tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def tokenize_and_align_labels(self, examples):
        tokenizer = self.tokenizer
        tokenized_inputs = tokenizer(examples["tokens"],
                                     truncation=False,
                                     is_split_into_words=True,
                                     padding="max_length",
                                     max_length=256)
        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(label[word_idx])
                previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def tokenize_dataset(self):
        tokenized_datasets = self.dataset.map(self.tokenize_and_align_labels,
                                              batched=True)
        return tokenized_datasets
