# Text2TTP

Cyber Threat Intelligence Report to MITRE ATT&CK Metrix

#### Core Maintainer and Developer: Udesh Kumarasinghe (mail @ udesh . xyz) 

## Directory Overview

- `data` contains the datasets created in this work.
  - `sentences.csv` - Aggregated threat behavior dataset.
  - `sentences_ioc.csv` - Sentence dataset annotated with IOCs.
  - `sentences_santitized.csv` - Sentence dataset with IOCs sanitized.
- `libs` - Python packages to load, preprocess, run the pipeline, and evaluate.
- `preprocessing` - Notebooks demonstrating the preprocessing steps.
- `models` - Our Pre-trained models used in our experiments. They are available on HuggingFace: [Models](https://huggingface.co/collections/qcri-cs/text2ttp-6648e136db255eefa7456d96)
- `Pipeline.ipynb` - Usage of proposed threat detection pipeline.

## How to Use

Refer to the `environment.yml` file to install the python packages required.

### Using the dataset
Load the aggregated dataset using the `resources` module. 

```python
from libs import resources as res

# Import all the sentences in the aggregated dataset
sentences = res.load_annotated()

# Filter to get the sentences of specific dataset
# Options available:
# 'manual' - manually annotated sentences in this work
# 'tram' - sentences from the training dataset of TRAM
# 'cisa' - annotated sentences extracted from CISA reports
# 'eset' - annotated sentences extracted from WeLiveSecurity reports
man_sentences = sentences[sentences.datasets == 'manual']
```

### Running the pipeline

Our implementation of the proposed pipeline uses the `pygaggle` framework. For 
simplicity, we exposed the `rank` module with functionality to preprocess queries,
initialize re-rankers and caching.

```python
from libs import rank

# Preprocess the MITRE ATT&K Knowledge Base and report sentences
texts, _ = rank.get_texts(corpus)
queries = rank.get_queries(sentences, label_col='tech_id')

# Initialize the reranking models for the pipeline
stage1_reranker = rank.construct_bm25()
stage2_reranker = rank.construct_sentsecbert()
stage3_reranker = rank.construct_monot5()
```

Refer to the `Pipeline.ipynb` for detailed example of the pipeline. Additional,
resources on how to run can be found at [here](https://github.com/castorini/pygaggle?tab=readme-ov-file#a-simple-reranking-example).

## Citation
```
@article{kumarasinghe2024semantic,
  title={Semantic Ranking for Automated Adversarial Technique Annotation in Security Text},
  author={Kumarasinghe, Udesh and Lekssays, Ahmed and Sencar, Husrev Taha and Boughorbel, Sabri and Elvitigala, Charitha and Nakov, Preslav},
  journal={arXiv preprint arXiv:2403.17068},
  year={2024}
}
```

## Third Party Frameworks

This work utilizes modified and extended versions of the following open-source works.

**PyGaggle** (https://github.com/castorini/pygaggle) - Used as a framework to build the re-ranking pipeline <br>
**ioc_parser** (https://github.com/armbues/ioc_parser) - For parsing Indicators of Compromise.
