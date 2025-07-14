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

## Experiments

## Results

| **Stanford Sentiment Treebank (SST)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |x%           |...            |

| **Quora Question Pairs (QQP)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |x%           |...            |

| **Semantic Textual Similarity (STS)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |x%           |...            |

| **Paraphrase Type Detection (PTD)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |x%           |...            |

| **Paraphrase Type Generation (PTG)** | **Metric 1** |**Metric n** |
|----------------|-----------|------- |
|Baseline |x%           |...            |

## Hyperparameter Optimization

## Visualizations 

## Members Contribution
### Jakob Faust
- Implemented `attention` and the `forward` method in `bert.py` 
- Implemented the Paraphrase type detection task (worked on the `bart_detection.py` file)
- Worked on the Optimizer.py together with Lukas Nölke, Franziska Ahme and Lennart Hahner.
- Worked together with Franziska Ahme and Lukas Nölke to merge and refactor code.  

### Lukas Nölke
- Implement `embed` in `bert.py`
- Implement Paraphrase detection in `multitask_classifier.py` (implement `predict_paraphrase`, implement data loading of the qqp dataset, implement training loop for the qqp task, implement paraphrase classifier)
- Implement `forward` methode in `multitask_classifier.py` together with Franziska Ahme.
- Fix wrong sperator in `dataset.py` together with Franziska Ahme.
- Implement dropout layer in `multitask_classifier.py` with Franziska Ahme.
- Worked on the Optimizer.py together with Jakob Faust, Franziska Ahme and Lennart Hahner.
- Worked together with Franziska Ahme and Jakob Faust to merge and refactor code.

### Franziska Ahme
- Implement `add_norm` in `bert.py`
- Implenent Sentiment Analysis in `multitasl_classifier.py` (implement `predict sentiment`, implement sentiment classifier)
- Implement `forward` methode in `multitask_classifier.py` together with Lukas Nölke.
- Fix wrong sperator in `dataset.py` together with Lukas Nölke.
- Implement dropout layer in `multitask_classifier.py` with Lukas Nölke.
- Worked on the Optimizer.py together with Jakob Faust, Lukas Nölke and Lennart Hahner.
- Worked together with Jakob Faust and Lukas Nölke to merge and refactor code.

### Lennart Hahner
- Implement Similarity Analysis in `multitask_classifier.py` (implement `predict_similarity`)
- Implement `forward` method in `multitaks_classifier.py`.
- Fix wrong seperator in `dataset`.
- Worked on the Optimizer.py together with Jakob Faust, Lukas Nölke and Lennart Hahner.
- Worked together with Jakob Faust and Lukas Nölke to merge and refactor code.

### Fabian Kathe



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