---

## NER Model Using BERT - Report

| Name of member | BITS ID | Email |
|---|---|---|
| Shreyas V | 2020B2A71350G | f20201350@goa.bits-pilani.ac.in |
| Pranav Srikanth | 2020B5A71865G | f20201865@goa.bits-pilani.ac.in |

### 1. Data Preparation and Pre-processing

Data for Named Entity Recognition (NER) was sourced from the CONLL2003 dataset. The dataset consists of sentences annotated with entity types such as person, organization, location, etc.

#### Pre-processing Steps:
- **Tokenization**: Used BERT's tokenizer to convert words into tokens.
- **Alignment**: Ensured that labels for NER are aligned with BERT's tokenization output.
- **Batching**: Organized data into batches for efficient training.
- **Data Splitting**: Divided the dataset into training, validation, and test sets.

### 2. Ground Truth

The ground truth for the CONLL2003 dataset comprises annotations for named entities in four categories: Person (PER), Organization (ORG), Location (LOC), and Miscellaneous (MISC).

### 3. Network Details

#### Model Architecture:
- **Base Model**: Utilized BERT's pre-trained model.
- **Token Classification Layer**: Added on top of BERT for NER task.
- **Modifications**: Varying number of layers, attention heads, and hidden units in experiments.

#### Training Details:
- **Optimizers**: Experimented with AdamW, Adam, SGD, etc.
- **Loss Functions**: Default cross-entropy loss, with exploration of alternatives in experiments.

### 4. Results

#### a. Gradient Check Results
- **Procedure**: Conducted gradient checks to ensure proper backpropagation.
- **Findings**: Verified the gradients are non-zero and decreasing over epochs, indicating effective learning.

#### b. Performance with Varying Parameters
- **Layer Variation**: Explored the impact of different numbers of BERT layers on the NER task.
- **Attention Heads**: Analyzed the model performance with varying numbers of attention heads.
- **Hidden Units**: Evaluated how changing the size of hidden units affects the results.

#### c. Extrinsic and Intrinsic Evaluation
- **Intrinsic**: Focused on model's ability to correctly identify and classify named entities within the dataset.
- **Extrinsic**: Applied the model to an external, real-world dataset to evaluate its generalization capabilities.

#### d. Evaluation Measures
- **Metrics Used**: Precision, Recall, and F1-Score.
- **Comparison**: Benchmarked against baseline models to illustrate improvements or changes.

---
