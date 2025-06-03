# `patent_novelty` Project

A Python pipeline for detecting **novelty**, **surprise**, and **distributional divergence** in patent texts. Combines natural language processing, statistical modeling, and unsupervised learning for analyzing innovation patterns over time.

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
