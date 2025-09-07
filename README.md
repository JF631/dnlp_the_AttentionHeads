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
---
## Methodology

### Paraphrase Detection - Quora Question Pairs (QQP)
To improve the existing baseline of the Quora Paraphrase Detection task, we focussed mainly on changing the architecture and introducing stronger regularization. Some additional ideas have been explored, but were ultimately discarded as they did not yield significant performance gains.

#### Architectural Changes ([5c9e7f1a](https://github.com/JF631/dnlp_the_AttentionHeads/pull/6/commits/5c9e7f1a61492ba0ff3712b3d59d1c6d66a2a7b2))
The initial implementation (baseline) of Paraphrase Detection Task with the QQP Dataset used a dual-encoder architecture, where each sentence in a pair is tokenized and processed individually. This is not only prevents rich token-level interactions between the two sentences, but also more computationally expensive, since two passes through the model are required per sentence pair.

To address those issues, the first improvement focussed on implementing a Single-Pass Cross-Encoder architecture, as introduced in [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805). This architectural change involves concatenating the two sentences and feeding them through the model in one forward pass, while using token type ids to to differentiate between the original sentences. This approach is not only more computationally efficient, but also improves the accuracy, since the self-attention layer can now model relationships between the two sentences.

#### Regularization ([6de22794](https://github.com/JF631/dnlp_the_AttentionHeads/pull/6/commits/6de22794fe30ca8ed97a52e1a97f5fef92889270))
Large pre-trained models like BERT are prone to overfitting, when training on smaller, task-specific datasets. This was also the case for the Paraphrase Detection task, using the QQP dataset, where the model would overfit after fine-tuning it for even a single epoch.

In order to try and combat overfitting, [R-Drop Regularization](https://arxiv.org/pdf/2106.14448) was introduced. The model now performs two forward passes with different dropout masks and calculates their averaged standard Cross-Entropy loss against the ground-truth labels. The KL-Divergence loss is then used to penalize the model if the two outputs are significantly different, forcing the two dropout sub-models to be consistent with each other.

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


### Semantic Textual Similarity (STS)

For STS and multitask improvements to BERT, I primarily consulted *“Enhancing miniBERT: Exploring Methods to Improve BERT Performance”* by Salehi et al. ([link](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/RajVPabari.pdf)) and implemented the highest-impact techniques according to their Table 3 results. Based on those findings, I implemented multitask fine-tuning and Mixed Attention.

#### Multitask Training

Multitask learning trains on multiple related tasks so the model can learn shared, more generalizable representations. By leveraging common structure across tasks, performance on each individual task can improve. Following Salehi et al. ([link](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/RajVPabari.pdf)), I implemented a multitask setup and addressed typical optimization issues with gradient-conflict methods (below).

**PcGrad Optimizer (`pcgradOptimizer.py`)**

PcGrad ([arXiv:2001.06782](https://doi.org/10.48550/arXiv.2001.06782)) adapts standard optimizers to multitask settings by detecting conflicting gradients (e.g., negative cosine similarity across tasks) and projecting them to reduce interference. This mitigates destructive updates between tasks and stabilizes learning.

**GradVac Optimizer (`gradvacOptimizer.py`)**

GradVac ([arXiv:2010.05874](https://doi.org/10.48550/arXiv.2010.05874)) also targets gradient interference but regularizes gradients toward agreement more proactively, not only when conflicts are detected. Conceptually, it enforces cross-task gradient alignment to promote stable, cooperative updates.

**SimBERT Model (`simBert.py`)**

Instead of a plain BERT cross-encoder, I implemented a Siamese/Triplet architecture (SBERT-style) for efficient similarity search, clustering, and retrieval ([arXiv:1908.10084](https://doi.org/10.48550/arXiv.1908.10084)). Cross-encoders score sentence pairs jointly—accurate but expensive at retrieval time. A Siamese encoder computes a fixed embedding for each sentence once, enabling fast approximate nearest-neighbor search—crucial in a multitask setup with larger datasets. As a rule of thumb, SBERT converts tens of millions of pairwise scores into one embedding per sentence plus inexpensive vector search.

Overall, these changes should improve generalization and scalability for STS within a multitask training pipeline.

#### Multi-Attention Layers 
**ConvBERT (`convBert.py`)**

Beyond the multitask setup, I implemented a mixed-attention architecture using ConvBERT ([arXiv:2008.02496](https://doi.org/10.48550/arXiv.2008.02496)). ConvBERT introduces span-based dynamic convolution that conditions the kernel on a local span, improving phrase-level disambiguation and paraphrase alignment. The mixed attention block retains some self-attention heads (to capture global structure and long-range cues like negation/quantifiers) while replacing redundant local heads with the span convolution (for precise n-gram/phrase alignment). In practice, this can map inputs to a lower-dimensional subspace within attention and reduce the number of heads, trimming computation without hurting accuracy.

#### Mean-Pooling instead of CLS-based

For sentence representations, I replace CLS-based embeddings with attention-masked mean pooling over the last hidden layer—an approach supported by SBERT and subsequent STS practice—yielding more robust similarity embeddings without requiring additional contrastive pretraining ([arXiv:1806.09828](https://doi.org/10.48550/arXiv.1806.09828)).

---
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

For QQP, run this command :
```sh
python multitask_classifier.py --option finetune --task=qqp --use_gpu --rdrop_alpha=2.0 --grad_clip=1.0
```

For SST, run this command : 
```sh

python multitask_classifier.py --option finetune --task=sst --use_gpu --local_files_only --seed [seed] --sst_pooling [cls, mean, max, attention] --sst_augmentation [synonym, backtranslation, none]
```

For STS, run this command:

```sh
python multitask_classifierSTS.py \
  --task sts \
  --model convBert \
  --hf_model_name YituTech/conv-bert-base \
  --option finetune \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --hidden_dropout_prob 0.1 \
  --batch_size 32 \
  --epochs 10 \
  --use_gpu \
  --amp
```
---
## Experiments

### Paraphrase Detection - Quora Question Pairs (QQP)
The initial architectural improvement, where we moved from the initial dual-encoder to the single-pass cross-encoder, provided the most significant improvement in accuracy, while also improving the training time.
Improvement/Version | Accuracy | F1 Score
-- | -- | --
Baseline | 0.771 | –
Cross-Encoder Encoding | 0.888 | 0.848

These results confirmed that allowing the model to process both sentences together is critical for the QQP task, thus making it the new base-version for all subsequent experiments.

Building on top of the now strong cross-encoder version, we experimented with several common techniques, incrementally adding them to the model. The main goal was to improve generalization and combat overfitting as well as the class imbalance of the QQP dataset. The results below were achieved with the default bert settings defined in `multitask_classifier.py` while training for only three epochs due to the lack of improvement and overall computation time and cost of the task.
Improvement/Version | Accuracy | F1 Score
-- | -- | --
Cross-Encoder Encoding (new Baseline) | 0.888 | 0.848
+[Pos_Weight](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) + [Label Smoothing](https://arxiv.org/pdf/1906.02629) | 0.885 | 0.854
+[Pair-order Augmentation](https://arxiv.org/pdf/1901.11196) (implemented as online augmentation)| 0.884 | 0.851
+Stronger Head | 0.884 | 0.851
+[Gradual Layer Unfreezing](https://arxiv.org/pdf/1801.06146) | 0.876 | 0.846
+[Mean Pooling](https://arxiv.org/pdf/1908.10084) | 0.884 | 0.840

However, as seen in the results above, none of the additions resulted in a clear and significant improvement. While some techniques provided a marginal increase in the F1 score, this often came at the cost of the accuracy. Due to the lack of consistent improvement, this development path was discarded and the model was [reverted to the single-pass encoder](https://github.com/JF631/dnlp_the_AttentionHeads/pull/6/commits/dae04b3ee7857532f99745158b3488df531a63b0).

Going back to the initial observation that the model was overfitting quickly, we decided to test a different regularization technique. After implementing R-Drop on top of the single-pass encoder baseline, we then focused on experimenting with different values for the hyperparameter `alpha`, as shown below.
Improvement/Version| Accuracy | F1
-- | -- | --
Cross-Encoder baseline (new Baseline) | 0.888 | 0.848
R-Drop (α=0.5) | 0.892 | 0.854
R-Drop (α=1.0) | 0.892 | 0.852
**R-Drop (α=2.0)** | **0.896** | **0.861**

We found that a `alpha=2.0` yielded the best overall performance on top of the cross-encoder baseline.

### Paraphrase type detection with BART.
The main drawback we noticed is the already very high accuracy score of the baseline model (around 90%). When we look at how the accuracy is computed however, we see that it is just the overlap between the ground truth multi hot vector and the predicted vector.
This means if the model "learns" to predict either always only zeroes or only the most frequent type in the training dataset the accuracy will be quite high even though the model is incapable of differentiating between 26 paraphrase types.
That the model learns to predict always the most frequent type comes from the fact, that the dataset used in training is unbalanced. E.g. we have several thousand examples of some paraphrase types on the one hand and only less than 10 examples of other types.

- we started off with a MultiLabelBalancedBatchSampler which oversamples rare labels via inverse-frequency probabilities so each batch includes more rare types.
- Next, we combined this with an [Asymmetric Loss For Multi-Label Classification](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf?) to. [reduce easy-negative dominance](https://arxiv.org/pdf/2507.11384).
- As this didn't improve the overall performance significantly, we introduced another loss term, the [Supervised Contrastive Loss](https://arxiv.org/pdf/2004.11362) to make the model cluster common paraphrase types together in the embedding dimension. 
- Additionally we introduced a nonlinear classification head which is already [discussed to perform better in BERT Models](https://arxiv.org/html/2403.18547v1) 

### Semantic Similarity Prediction - Semantic Textual Similarity (STS)
#### Multitask-Training 

I first implemented a basic multitask setup. 
Multitask training should improve BERT’s generalization and, in turn, STS performance. 
Below are concise results for several multitask variants. Optimizer choices target the well-known issue 
of **conflicting gradients** in multitask learning, as discussed by Femrite 
(“Rhapsody on a Theme of Gradient Surgery”) and formalized in PcGrad 
([arXiv:2001.06782](https://doi.org/10.48550/arXiv.2001.06782)) and 
GradVac ([arXiv:2010.05874](https://doi.org/10.48550/arXiv.2010.05874)). 
I also tested an SBERT-style Siamese model 
([arXiv:1908.10084](https://doi.org/10.48550/arXiv.1908.10084)).

Metrics:
- QQP (Quora Question Pairs): accuracy  
- SST: accuracy  
- STS: Dev Pearson correlation

| Variant               | QQP Acc | SST Acc | STS ρ |
|-----------------------|:-------:|:-------:|:-----:|
| Vanilla Multitask     | 0.784   | 0.521   | 0.304 |
| PcGrad ([link](https://doi.org/10.48550/arXiv.2001.06782))   | 0.784   | 0.524   | 0.282 |
| GradVac ([link](https://doi.org/10.48550/arXiv.2010.05874))  | 0.784   | 0.524   | 0.282 |
| SBERT + PcGrad ([link](https://doi.org/10.48550/arXiv.1908.10084)) | 0.770   | 0.509   | 0.326 |

Takeaways:
- Gradient-surgery methods (PcGrad/GradVac) did **not** improve STS; SST/QQP remained flat.
- SBERT-style Siamese training slightly recovers STS (0.326) but lags on QQP/SST versus vanilla.

Hyperparameters used:
```
{'b:watch_size': 64,
 'epochs': 10,
 'filepath': 'models/finetune-10-1e-05-multitask.pt',
 'hidden_dropout_prob': 0.1,
 'local_files_only': False,
 'lr': 1e-05,
 'model': 'simBert',
 'optimizer': 'pcgrad',
 'option': 'finetune',
 'seed': 11711,
 'task': 'multitask',
 'use_gpu': True}
```

To reproduce:

```sh
python multitask_classifierMultitask.py \ 
 --use_gpu \
 --model simBert \
 --optimizer pcgrad \
 --task multitask \
 --option finetune \
 --hidden_dropout_prob 0.1
```

#### Multilayer-Attention 

For STS alone, I switched to a mixed-attention encoder using ConvBERT: *“ConvBERT: Improving BERT with Span-based Dynamic Convolution”* ([arXiv:2008.02496](https://doi.org/10.48550/arXiv.2008.02496)). ConvBERT replaces some local self-attention heads with span-based dynamic convolutions while keeping global heads, which should help align phrase-level semantics without losing long-range cues.

Metrics: 
- STS: Dev Pearson correlation

| Task | Metric                  | Value |
|------|-------------------------|:-----:|
| STS  | Dev Pearson correlation | 0.338 |

Training/engineering fixes applied:

- Load a compatible pretrained backbone: `YituTech/conv-bert-base`.
- Enable full encoder fine-tuning (HF backbone).
- Optimizer: Hugging Face AdamW with decoupled weight decay and parameter groups.
- Learning-rate schedule with warmup; gradient clipping for stability.
- Dataloader and `datasets.py` stability fixes.

Result after:

| Task | Metric                  | Value |
|------|-------------------------|:-----:|
| STS  | Dev Pearson correlation | 0.397 |

Hyperparameters used:
```json
{
  "amp": true,
  "batch_size": 32,
  "epochs": 10,
  "filepath": "models/finetune-10-2e-05-sts.pt",
  "hf_model_name": "YituTech/conv-bert-base",
  "hidden_dropout_prob": 0.1,
  "local_files_only": false,
  "lr": 2e-05,
  "max_grad_norm": 1.0,
  "model": "convBert",
  "optimizer": "adamw",
  "option": "finetune",
  "seed": 11711,
  "task": "sts",
  "use_gpu": true,
  "warmup_ratio": 0.1,
  "weight_decay": 0.01
}
```
To reproduce:

```sh
python multitask_classifierSTS.py \
  --task sts \
  --model convBert \
  --hf_model_name YituTech/conv-bert-base \
  --option finetune \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --hidden_dropout_prob 0.1 \
  --batch_size 32 \
  --epochs 10 \
  --use_gpu \
  --amp
```

#### Mean-Pooling used with of Attention-mask
Replace the function `_encode` in multitask_classifierSTS with

```python
def _encode(self, input_ids, attention_mask):
    out = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    last = out.last_hidden_state                   
    mask = attention_mask.unsqueeze(-1)           
    summed = (last * mask).sum(dim=1)             
    denom = mask.sum(dim=1).clamp(min=1)          
    return summed / denom
```
Resulting in:

| Task | Metric                  | Value |
|------|-------------------------|:-----:|
| STS  | Dev Pearson correlation | 0.395 |

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
### Improvements
#### Multitask Training using simBert and pcGrad
| Variant               | QQP Acc | SST Acc | STS ρ |
|-----------------------|:-------:|:-------:|:-----:|
| Vanilla Multitask     | 0.784   | 0.521   | 0.304 |
| PcGrad ([link](https://doi.org/10.48550/arXiv.2001.06782))   | 0.784   | 0.524   | 0.282 |
| GradVac ([link](https://doi.org/10.48550/arXiv.2010.05874))  | 0.784   | 0.524   | 0.282 |
| SBERT + PcGrad ([link](https://doi.org/10.48550/arXiv.1908.10084)) | 0.770   | 0.509   | 0.326 |


#### Semantic Textual Similarity using convGrad
| Task | Metric                  | Value |
|------|-------------------------|:-----:|
| STS  | Dev Pearson correlation | 0.397 |

#### Sentiment Analysis

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

### Baseline

| **Quora Question Pairs (QQP)** | **Dev Accuracy** | **Dev F1 Score**
|-----------|-----------|-----------|
|Baseline | 0.781 | -
Cross-Encoder| 0.888 | 0.848
~~+Pos_Weight + Label Smoothing~~ | ~~0.885~~ | ~~0.854~~
~~+Pair-order Augmentation (implemented as online augmentation)~~| ~~0.884~~ | ~~0.851~~
~~+Stronger Head~~ | ~~0.884~~ | ~~0.851~~
~~+Gradual Layer Unfreezing~~ | ~~0.876~~ | ~~0.846~~
~~+Mean Pooling~~ | ~~0.884~~ | ~~0.840~~
**R-Drop (α=2.0)** | **0.896** | **0.861**

Note: The crossed out imprvementes/versions have been discarded/reverted due to the lack of significant improved results.

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

---

## Hyperparameter Optimization

---

## Visualizations 

---

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
- Improve baseline of the Quora Praphrase Detection task:
  - Research and implement cross-encoder architecture
  - Research and implement pos_weight and label smoothing (discarded)
  - Research and implement online pair order augmentation (discarded)
  - Implement stronger head (discarded)
  - Research and implement gradual layer unfreezing (discarded)
  - Research and implement mean pooling (discarded)
  - Research and implement R-Drop
- Merge and assist in merging different task branches into main branch with all other group members

### Franziska Ahme
- Implement `add_norm` in `bert.py`
- Implement Sentiment Analysis in `multitask_classifier.py` (implement `predict_sentiment`, implement sentiment classifier)
- Implement `forward` method in `multitask_classifier.py` together with Lennart Hahner and Lukas Nölke
- Implement dropout layer in `multitask_classifier.py` with Lukas Nölke
- Research and implement different pooling methods (mean, max and attention pooling)
- Research and implement data augmentation methods (synonym replacement and backtranslation)

### Lennart Hahner
- Implement Similarity Analysis in `multitask_classifier.py` (implement `predict_similarity`)
- Implement `forward.py` method in `multitaks_classifier.py` with Franziska Ahme and Lukas Nölke
- Implement `simBert.py` for Multitask training using Task-specific layers and enhancing performance.
- Implement `convBert.py` for Multi-Attention Layers introducing better local attention for STS.
- Implement `datasetsSTS.py`, `evaluationSTS.py` and `bertSTS.py` to tailor script more for convBert and to help Lukas Nölke and Franziska Ahme to keep their results without conflicts.
- Implement `gradvacOptimizer.py` as optimizer for multitask training.
- Implement `pcgradOptimizer.py` as alternative optimizer for multitask training.
- Implement `multitask_classifierMultitask.py` to enable multitask training by excluding the ETPC dataset.
- Implement `buildSimBertFromHF.by` to load pre-trained model simBert from Huggingface and use it for finetuning SimBert.

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
