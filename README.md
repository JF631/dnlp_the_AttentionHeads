# The Attention Heads - DNLP SS25 Final Project
- Group repository: [Current repo](https://github.com/JF631/dnlp_the_AttentionHeads)
- Tutor Responsible: Corinna Wegner
- Group team leader: Jakob Faust
- Group members: Lukas Nölke, Franziska Ahme, Lennart Hahner, Fabian Kathe, Jakob Faust


## Setup instructions
* Make sure you have conda installed on your system.
* Follow `setup.sh` to properly setup a conda environment and install dependencies.
* For training the model on GWDG's GPU cluster use `setup_gwdg.sh` instead.
* All packages that are needed will be installed in a conda environment called `dnlp` 

## Methodology

### Sentiment Analysis - Stanford Sentiment Treebank (SST)

**Pooling Methods**

To go beyond BERT's standard approach using the [CLS] token embedding as a summary of the input sequence, we experimented with alternative pooling methods: mean pooling, max pooling and attention-based pooling. 
- Mean pooling: Mean pooling computes the average of all token embeddings in the sequence. This could result in a small improvement by capturing information from all tokens, potentially improving sentence-level representation.
- Max pooling: Max pooling takes the maximum value across all tokens.
- Attention pooling: Attention pooling learns a weighted combination of token embeddings. This allows the model to focus on the most relevant parts of the input. We expected this to have the highest rise in accuracy.

**Data augmentation**

Training NLP models often benefit from larger datasets. The original SST dataset is quite small, so augmentation can improve generalization and robustness and overcome the limitation of small training data.

- Synonym replacement: In this approach we randomly replace a fixed number of words (10%, 25%, 50%) of a sentence with their synonyms (from WordNet) to create more training data. This replacement introduces lexical variety to the train dataset without changing the overall sentiment. As reference, we used [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks - Jason Wei, Kai Zou](https://arxiv.org/abs/1901.11196).
- Backtranslation: This approach generates semantically equivalent paraphrases via machine translation, in this case Hugging Face’s MarianMT models. This approach translates an English sentence into French and then back into English, producing natural paraphrases, that are not in the training data exactly as is. The idea for this approach come from [Improving Neural Machine Translation Models with Monolingual Data - Rico Sennrich, Barry Haddow, Alexandra Birch](https://arxiv.org/abs/1511.06709).

## Training

#### Part 1: Baseline
To train the model and reproduce our results for the baseline, run these commands after activating the environment

For SST, QQP, STS:

```sh

python multitask_classifier.py --option finetune --task=[sst, sts, qqp] --use_gpu --local_files_only
```
**Hint**: To make it run for the required subtasks we changed the name of the "etpc_dev" dataset in the code to the test-data for etpc "etpc-paraphrase-detection-test-student".

For Paraphrase Type Detection:
```sh
python bart_detection.py --use_gpu --seed 1171
```

## Experiments

### Sentiment Analysis - Stanford Sentiment Treebank (SST)
We ran a series of experiments to evaluate the effect of pooling methods and data augmentation on classification performance on SST. All experiments were repeated with n=4 different seeds (11711, 42, 2025, 34567) to account for variance in training. Each model was finetuned with the 4 pooling methods: [CLS] token embedding (baseline), mean pooling, max pooling and attention pooling. To test the effect of dataset expansion we applied synonym replacement and backtranslation, each was run 4 times using [CLS] pooling.

The following additional packages are required for reproducibility: pip install nlpaug, pip install nltk

Note: 
- nltk_data/ is included in the repository, and no extra download is needed,
- A precomputed backtranslated dataset (sst_backtranslated.json) is also in ther epository. This avoids rerunning with MarianMT.

The experiments were run with: 
```sh

python multitask_classifier.py --option finetune --task=sst --use_gpu --local_files_only --seed [seed]
```

## Results

| **Stanford Sentiment Treebank (SST)** | **Dev Accuracy** |
|----------------|-----------|
|Baseline |0.535 (0.006)           |      
|Mean Pooling |0.542 (0.006)           |
|Max Pooling |0.537 (0.004)           |
|Attention Pooling |0.543 (0.008)           |
|Synonym replacement (aug_p=0.1) |0.532 (0.007)           |
|Synonym replacement (aug_p=0.25) |0.534 (0.002)           |
|Synonym replacement (aug_p=0.5) |0.529 (0.008)           |
|Backtranslation |0.541 (0.004)           |



| **Quora Question Pairs (QQP)** | **Dev Accuracy** |
|----------------|-----------|
|Baseline |0.781 (78.1%)          |

| **Semantic Textual Similarity (STS)** | **Dev Accuracy** |
|----------------|------------------|
|Baseline | 0.345(34.5%)               |

| **Paraphrase Type Detection (PTD)** | **Dev Accuracy** |**Matthews Correlation Coefficient (MCC)** |
|----------------|-----------|------- |
|Baseline |0.904 (90.4%)           | 0.102           |

| **Paraphrase Type Generation (PTG)** | BLEU Score |
|----------------|-----------|
|Baseline | 44.3   |

## Hyperparameter Optimization

## Visualizations 

## Members Contribution
### Jakob Faust
- Implement `attention` and the `forward` method in `bert.py` 
- Implement the Paraphrase type detection task (worked on the `bart_detection.py` file)
- Worked on the `optimizer.py`

### Lukas Nölke
- Implement `embed` in `bert.py`
- Implement Paraphrase detection in `multitask_classifier.py` (implement `predict_paraphrase`, implement data loading of the qqp dataset, implement training loop for the qqp task, implement paraphrase classifier)
- Implement `forward` methode in `multitask_classifier.py` together with Franziska Ahme and Lennart Hahner
- Implement dropout layer in `multitask_classifier.py` with Franziska Ahme

### Franziska Ahme
- Implement `add_norm` in `bert.py`
- Implement Sentiment Analysis in `multitask_classifier.py` (implement `predict_sentiment`, implement sentiment classifier)
- Implement `forward` method in `multitask_classifier.py` together with Lennart Hahner and Lukas Nölke
- Implement dropout layer in `multitask_classifier.py` with Lukas Nölke
- Research and implement different pooling methods (mean, max and attention pooling)
- Research and implement data augmentation methods (synonym replacement and backtranslation)

### Lennart Hahner
- Implement Similarity Analysis in `multitask_classifier.py` (implement `predict_similarity`)
- Implement `forward` method in `multitaks_classifier.py` with Franziska Ahme and Lukas Nölke

### Fabian Kathe
- Implemented context of empty pipeline with transform_data, train_model, test_model, evaluate_model, finetune_paraphrase_generation
- Uploaded chache with base model to Cluster, needed for finetuning and for local files only


# AI-Usage Card

# References 

### Acknowledgement

The project description, partial implementation, and scripts were adapted from the default final project for the Stanford [CS 224N class](https://web.stanford.edu/class/cs224n/) developed by Gabriel Poesia, John, Hewitt, Amelie Byun, John Cho, and their (large) team (Thank you!)

The BERT implementation part of the project was adapted from the "minbert" assignment developed at Carnegie Mellon University's [CS11-711 Advanced NLP](http://phontron.com/class/anlp2021/index.html),
created by Shuyan Zhou, Zhengbao Jiang, Ritam Dutt, Brendon Boldt, Aditya Veerubhotla, and Graham Neubig  (Thank you!)

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers) library ([Apache License 2.0](./LICENSE)).

Parts of the scripts and code were altered by [Jan Philip Wahle](https://jpwahle.com/) and [Terry Ruas](https://terryruas.com/).

The project was modified by [Niklas Bauer](https://github.com/ItsNiklas/) for the 2025 DNLP course at the University of Göttingen.


## Contributing
