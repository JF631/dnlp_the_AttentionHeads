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

### Paraphrase type detection with BART
To further finetune BART to differentiate between 26 paraphrase types, we focussed on the overall training on inabalanced data sets.
Therefore, we considered following strategies:
- New sampling mehtod inside training batches in a way that more rare types are over represented in batches.
- This can be combined with an [Asymmetric Loss (ASL) for Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf?) to focus more on rare labels during traing. This also tackles the easy-negative dominance problem.
Here, negative dominance describes that the 26 dimensional multi hot encoded vector of {0, 1} has overall much more zeros than ones, as one sentence pair only represents some few paraphrase types (or none at all).
- We also considered [class weighting methods](https://arxiv.org/pdf/2507.11384) but sticked with frequency based over sampling of rare labels, as it is more robust.
- [Supervised Contrastive Loss](https://arxiv.org/pdf/2004.11362) to make the model cluster common paraphrase types together in the embedding space. By this, it is easier for the model to actually gain a language understanding and not just learn to predict the most common paraphrase type.
- A small MLP as classification head instead of the linear head so far. This has often been discussed in papers, especially in combination with BERT.(e.g., [[1](https://arxiv.org/html/2403.18547v1)], [[2](https://arxiv.org/abs/2210.16771)])


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

#### Part 2: Improvements

For SST, run this command : 
```sh

python multitask_classifier.py --option finetune --task=sst --use_gpu --local_files_only --seed [seed] --sst_pooling [cls, mean, max, attention] --sst_augmentation [synonym, backtranslation, none]
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

python multitask_classifier.py --option finetune --task=sst --use_gpu --local_files_only --seed [seed] --sst_pooling [cls, mean, max, attention] --sst_augmentation [synonym, backtranslation, none]
```

Overall, the implemented improvements did not lead to the hoped increased accuracy and could not solve the problem of overfitting. 

**Pooling Methods**

Mean and attention pooling led to a small improvement in accuracy. Max pooling has very similar results as the baseline. 

**Data augmentation**

Synonym replacement worsened the accuracy, especially for aug_p=0.1 and aug_p=0.5. This was expected, since too little change in sentence variance or too large changes in meaning lead to reduced accuracy. For aug_p=0.25, we have similar results as for the baseline. 

Backtranslation managed to increase the accuracy consistently very marginally, but still the training accuracy goes way up, while dev accuracy stays flat at around 0.50–0.54, which means the model is essentially learning to memorize the augmented data as well but still fails to transfer to unseen examples.

### Paraphrase type detection with BART.
The main drawback we noticed is the already very high accuracy score of the baseline model (around 90%). When we look at how the accuracy is computed however, we see that it is just the overlap between the ground truth multi hot vector and the predicted vector.
This means if the model "learns" to predict either always only zeroes or only the most frequent type in the training dataset the accuracy will be quite high even though the model is incapable of differentiating between 26 paraphrase types.
The fact that the model learns to predict always the most frequent type arises beacuse the dataset used for training is unbalanced. E.g. we have several thousand examples of some paraphrase types on the one hand and only less than 10 examples of other types.

We mainly focussed on overcomming this major problem by taking and trying out following approaches:

- we started off with a MultiLabelBalancedBatchSampler which oversamples rare labels via inverse-frequency probabilities so that each batch includes more rare types.
- Next, we combined this with an ASL (see Methodology) to focus even more on rare types.
- As this didn't improve the overall performance significantly, we introduced Supervised Contrastive Learning  to make the model cluster common paraphrase types together in the embedding space to learn real relationships between paraphrase types. 
- Additionally, we introduced a nonlinear classification head which requires only slightly more effort to train.

The overall improved outcome is quite well summarized in these pictures: 
<img width="600" height="400" alt="per_label_f1_delta" src="https://github.com/user-attachments/assets/6e862f1e-71f0-4063-84b2-f6829468818b" />
<img width="600" height="400" alt="per_label_f1_before_after" src="https://github.com/user-attachments/assets/069c6302-d48b-4614-82cb-2deeb43db1ae" />


As can be seen, the model now predicts rare types much better than before and improves its recognition performance on almost all paraphrase types (measured by the F1 score) on the dev set.

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

| **Paraphrase Type Detection (PTD)** | **Dev Accuracy** |**Matthews Correlation Coefficient (MCC)** | **Average F1 scor (over all labels)** |
|----------------|-----------|------- | ------- |
|Baseline |0.904 (90.4%)           | 0.102           | 0.1915 |
|Data Loader |0.890 (89.0%)           | 0.080           | 0.1755|
|Nonlinear classifier + ASL |0.836 (83.6%)           | 0.099           | 0.2583|
|Supervised Contrastive Loss | 0.853 (85.3%) | 0.154 | 0.2993 |

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
