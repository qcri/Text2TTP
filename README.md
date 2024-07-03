# Text2TTP

Cyber Threat Intelligence Report to MITRE ATT&CK Metrix

This is the reproduction material for our work: "Semantic Ranking for Automated Adversarial Technique Annotation in Security Text" published in AsiaCCS'24.

If you use our tool, models, or dataset, please cite our work:

```
@inproceedings{10.1145/3634737.3645000,
    author = {Kumarasinghe, Udesh and Lekssays, Ahmed and Sencar, Husrev Taha and Boughorbel, Sabri and Elvitigala, Charitha and Nakov, Preslav},
    title = {Semantic Ranking for Automated Adversarial Technique Annotation in Security Text},
    year = {2024},
    isbn = {9798400704826},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3634737.3645000},
    doi = {10.1145/3634737.3645000},
    abstract = {We introduce a novel approach for mapping attack behaviors described in threat analysis reports to entries in an adversarial techniques knowledge base. Our method leverages a multi-stage ranking architecture to efficiently rank the most related techniques based on their semantic relevance to the input text. Each ranker in our pipeline uses a distinct design for text representation. To enhance relevance modeling, we leverage pretrained language models, which we fine-tune for the technique annotation task. While generic large language models are not yet capable of fully addressing this challenge, we obtain very promising results. We achieve a recall rate improvement of +35\% compared to the previous state-of-the-art results. We further create new public benchmark datasets for training and validating methods in this domain, which we release to the research community aiming to promote future research in this important direction.},
    booktitle = {Proceedings of the 19th ACM Asia Conference on Computer and Communications Security},
    pages = {49–62},
    numpages = {14},
    keywords = {threat intelligence, TTP annotation, text ranking, text attribution},
    location = {Singapore, Singapore},
    series = {ASIA CCS '24}
}
```

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

## Third Party Frameworks

This work utilizes modified and extended versions of the following open-source works.

**PyGaggle** (https://github.com/castorini/pygaggle) - Used as a framework to build the re-ranking pipeline <br>
**ioc_parser** (https://github.com/armbues/ioc_parser) - For parsing Indicators of Compromise.
