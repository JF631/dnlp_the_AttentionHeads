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

## Training

#### Part 1: Baseline

To train the model and reproduce our results for the baseline, run this command after activating the environment:

```sh

python multitask_classifier.py --option finetune --task=[sst, sts, qqp] --use_gpu --local_files_only

```

## Experiments

## Results

| **Stanford Sentiment Treebank (SST)** | **Dev Accuracy** |
|----------------|-----------|
|Baseline |0.530 (53.0%)           |

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
