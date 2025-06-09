## 🚀 Getting Started

To run the project locally:

1. Clone the repository, install dependencies, and set up folders  
   	```bash
   	git clone https://github.com/edgarLan/patent_novelty.git
   	cd patent_novelty
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	python create_structure.py
	```

2 - Download compressed patent data in /data/compressedData/
	Place .tar.gz files (for years 2007–2016) in

3 - Download citation data files.tsv files in /data/citationsData
	g_patent.tsv
	g_us_patent_citation.tsv

4 - Download .txt files in /data/vocab/vocab/additional stopwords
	technical_stopwords.txt
	USPTO_stopwords_en.txt

5 - Unzip app-patent match data
	Unzip and place contents in /data/app_pat_match/

6 - (Optional) Run demo instead of full project. Demo is only IPC=F03D, year=2016
If you prefer to skip full preprocessing:
	Unzip the csv_clean/
	Place the folders (ES/, KS/, tE/) into /data/csv_clean/
	in config.txt, steps to run, put true from metrics_raw onwards.
	forDemo=True in metrics computation runs metrics for 90 (or less) patents of toEval.
	
7 - Use config.txt to adjust inputs



# `patent_novelty` Project

A Python pipeline for detecting novelty (newness, uniqueness and difference), surprise in patent texts. Combines natural language processing, statistical modeling, and unsupervised learning for analyzing innovation patterns over time.

---

## 📦 Data Setup

### 1. Patent Compressed Data  
- **Source**: [HuggingFace HUPD Dataset](https://huggingface.co/datasets/HUPD/hupd/tree/main/data)  
- **Download**: `.tar.gz` files for selected years (2007–2016)

### 2. Vocabulary Files  
- **Sources**:  
  - [Technical Stopwords](https://github.com/SerhadS/TechNet/tree/master/vocabulary)  
  - [Additional Stopwords](https://github.com/SerhadS/TechNet/tree/master/additional_stopwords)  
- **Download**:
  - `technical_stopwords.txt`
  - `USPTO_stopwords_en.txt`

### 3. Citations Data  
- **Source**: [PatentsView Data Portal](https://patentsview.org/download/data-download-dictionary)  
- **Download**:  
  - `g_patent.tsv`  
  - `g_us_patent_citation.tsv`

> **Note**: Files marked with `*` in the structure below must be downloaded manually.

---

## 🗂️ Project Structure

```
patent_novelty/
├── metrics/
│   ├── metrics_raw/
│   ├── top10/
│   ├── cd_index/
│   ├── metrics/
│   └── metricAnalysis/
├── data/
│   ├── app_pat_match/
│   ├── citationsData/
│   │   ├── g_patent.tsv               * Download required
│   │   └── g_us_patent_citation.tsv  * Download required
│   ├── compressedData/               * .tar.gz files (2007–2016)
│   ├── csv_clean/
│   │   ├── ES/
│   │   ├── KS/
│   │   └── tE/
│   ├── csv_raw/
│   │   ├── ES/
│   │   │   └── text/
│   │   ├── KS/
│   │   └── tE/
│   ├── jsonData/
│   └── vocab/
│       ├── technet from github/
│       │   ├── vocab_github_1.tsv    * Download required
│       │   └── vocab_github_2.tsv    * Download required
│       └── additional stopwords/
│           ├── manual_adds.txt
│           ├── technical_stopwords.txt   * Download required
│           └── USPTO_stopwords_en.txt    * Download required

```

\* Files marked with an asterisk must be downloaded manually.

---

## 🧩 Module Descriptions

### `importation.py`  
Handles data extraction and transformation from `.tar.gz` archives to JSON and CSV formats.

### `cleaning.py`  
Provides tools for token cleaning, lemmatization, and stopword filtering based on vocabularies. Uses `spaCy`, `NLTK`, and `pandas`.

### `divergences.py`  
Implements Jensen-Shannon Divergence (JSD) computations for comparing probability distributions.

### `novelty.py`  
Defines classes for novelty detection:
- **Newness**: JSD-based divergence from reference
- **Uniqueness**: Distributional shift vs prototype
- **Difference**: Deviation from neighboring distributions
- **ClusterKS**: KMeans clustering for scalable divergence analysis

### `surprise.py`  
Analyzes co-occurrence pattern shifts using PMI (Pointwise Mutual Information) and surprise scoring.

### `utils.py`  
Utility functions for:
- Vocabulary and lemmatization pipelines
- PMI construction and update
- Metric evaluation and statistical testing

---

## 📊 Key Features

- **Supports longitudinal analysis** across multiple years and IPC classes  
- **Modular design** for extensibility and experimentation  
- **Built-in statistical validation** (t-tests, correlations, etc.)  
- **Compatible with sparse/dense representations** for scalable divergence measures  
- **Batch processing** with progress tracking for large corpora  

---

## ⚙️ Requirements

Recommended dependencies:
- `pandas`, `numpy`, `scipy`, `nltk`, `spacy`, `tqdm`, `sklearn`, `statsmodels`, `rbo`

To install via pip:
```bash
pip install -r requirements.txt
