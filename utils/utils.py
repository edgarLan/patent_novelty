from collections import Counter
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures
from nltk.tokenize import word_tokenize
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from collections import OrderedDict, defaultdict, Counter
import pandas as pd
import numpy as np
from nltk.probability import FreqDist
from tqdm import tqdm
import copy

import spacy
nlp = spacy.load("en_core_web_sm")
from nltk.corpus import stopwords
import os

import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_ind, kendalltau
from itertools import combinations
import numpy as np
import rbo  # Assuming rbo is already installed
import statsmodels.api as sm


from collections import Counter
from nltk.util import ngrams
import math
from tqdm import tqdm

from fast_cdindex import cdindex
import datetime

import json


from utils.cleaning import get_file_names, extract_year_ipc, extract_year_ipc_vs, extract_year_ipc_vs_top10
import os
from collections import defaultdict

# from utils.utils import docs_distribution, new_distribution, combine_columns, OptimizedIncrementalPMI
from novelty.novelty import Newness, Uniqueness, Difference, ClusterKS
from novelty.surprise import Surprise
import time

##### Technet #####

def createCleanTechnet(pathData):
    """
    Create and save a cleaned vocabulary list by splitting Technet keywords on underscores.

    Parameters:
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab). vocab contains vocab folder and technet from gihub folder containing the 'vocab_github_1.tsv' and 'vocab_github_2.tsv' files.

    Returns:
        None: Writes the cleaned vocabulary to 'clean_vocab.csv' in the specified folder.
    """
    pathVocab = os.path.join(pathData, "vocab", "vocab")
    pathVocabGit = os.path.join(pathData, "vocab", "technet from github")

    df_vocab_1 = pd.read_csv(os.path.join(pathVocabGit, 'vocab_github_1.tsv'), sep='\t', header=None)
    df_vocab_2 = pd.read_csv(os.path.join(pathVocabGit, 'vocab_github_2.tsv'), sep='\t', header=None)

    # List of keywords in Technet
    kw_list_1 = list(df_vocab_1[0])
    kw_list_2 = list(df_vocab_2[0]) 

    # Keep all individual words (all words separated by "_" are split)
    total_list = kw_list_1 + kw_list_2
    list_of_tokens = []
    for i in tqdm(range(len(total_list))):
        word = str(total_list[i])
        word_list = word.split('_')
        list_of_tokens.extend(word_list)
        
    final_list = list(set(list_of_tokens))

    df_expect = pd.DataFrame({'technet_vocab': final_list}).dropna()

    # remove empty or space words
    df_expect = df_expect[~df_expect['technet_vocab'].isin(['', ' '])]
    df_expect=df_expect.dropna(subset=['technet_vocab']).reset_index(drop=True)
    df_expect.to_csv(os.path.join(pathVocab, 'clean_technet.csv'), na_rep="NA", index=False)


def lemmatizeTechnet(pathData):
    """
    Lemmatizes the 'technet_vocab' column from a cleaned TechNet vocabulary file with spaCy.

    Parameters:
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab). vocab/vocab contains 'clean_technet.csv'.

    Returns:
        None: Saves the lemmatized vocabulary to 'lemmatized_technet.csv' in the same directory.
    """

    pathVocab = os.path.join(pathData, "vocab", "vocab")

    technet = pd.read_csv(os.path.join(pathVocab, 'clean_technet.csv'))
    technet_NA=technet.dropna(subset=['technet_vocab']).reset_index(drop=True)
    set_vocab = list(technet_NA['technet_vocab'])
    lemmatized_technet = []

    # Use nlp.pipe() for batch processing, ner and parser disabled because computationnaly extensive, and not useful for single words
    for doc in tqdm(nlp.pipe(set_vocab, disable=["ner", "parser"])):
        # Join token lemmas, preserving dashes/slashes if necessary
        lemmatized_word = "".join([token.lemma_ if token.is_alpha else token.text for token in doc])
        lemmatized_technet.append(lemmatized_word)

    df=pd.concat([pd.DataFrame(technet_NA), pd.DataFrame(lemmatized_technet, columns=["lemmatized"])], axis=1)

    df.to_csv(os.path.join(pathVocab, 'lemmatized_technet.csv'), index=False)





##### Surprise #####

def pmi(input_, w_size=3):
    """ Parameters : list of ORDERED feature variables (words for example) -- if feature order does not matter set window_size to inf. """
    bigram_measures = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(input_, window_size= w_size)
    return finder.score_ngrams(bigram_measures.pmi) #, finder.word_fd, finder.ngram_fd, total_words_temp

class OptimizedIncrementalPMI:
    """
    An optimized class for incrementally computing Pointwise Mutual Information (PMI) 
    between word pairs (bigrams) based on a sliding window.

    This class supports updating internal statistics with new input data.

    Attributes:
        window_size (int): Size of the context window used for bigram construction.
        word_counts (Counter): Counter object tracking individual word frequencies.
        bigram_counts (Counter): Counter object tracking bigram (word pair) frequencies.
        total_words (int): Total number of words processed so far.
    """

    def __init__(self, window_size=3, word_counts=Counter(), bigram_counts=Counter(), total_words=0, initial = True):
        """
        Initialize the PMI model.

        Args:
            window_size (int): Size of the sliding window to use when forming bigrams. Default is 3.
            word_counts (Counter): (Optional) Pre-existing word frequency counts.
            bigram_counts (Counter): (Optional) Pre-existing bigram frequency counts.
            total_words (int): (Optional) Total number of words seen in previous runs.
            initial (bool): Whether to initialize from scratch (True) or use provided counters (False).
        """
        if initial:
            self.window_size = window_size
            self.word_counts = Counter()  # Word frequency counts
            self.bigram_counts = Counter()  # Bigram counts
            self.total_words = 0  # Total number of words seen so far`
        else:
            self.window_size = window_size
            self.word_counts = word_counts  
            self.bigram_counts = bigram_counts
            self.total_words = total_words

    def update(self, input_):
        """
        Update the model with a new sequence of words.

        This updates both word and bigram counts (self) using a sliding window.

        Args:
            input_ (List[str]): A list of tokens (words) from a new document or sentence.
        """
        # Update word counts
        self.word_counts.update(input_)
        self.total_words += len(input_)

        # Generate bigrams for each window size
        bigrams = (
            (input_[i], input_[j])
            for i in range(len(input_))
            for j in range(i + 1, min(i + self.window_size, len(input_)))
        )
        self.bigram_counts.update(bigrams)

    
    def compute_pmi(self):
        """
        Compute the Pointwise Mutual Information (PMI) for all observed bigrams.

        Returns:
            dict: A dictionary mapping (word1, word2) tuples to their PMI scores.
        """
        pmi_scores = {}

        # Logarithmic PMI formula (log2 version)
        for (word1, word2), count in (self.bigram_counts.items()):
            pmi = math.log2(count/(self.word_counts[word1]*self.word_counts[word2])) + math.log2(self.total_words*1/(self.window_size - 1))
            pmi_scores[(word1, word2)] = pmi 
        return pmi_scores 
    
    

def pmi_to_dict_adj(pmi_list):
        """ Take a PMI list of tuples in nltk format [((w1,w2),value)] and output a nested dictionary """
        nested_dict = {}
        w1 = list(OrderedDict.fromkeys(item[0][0] for item in pmi_list))
        w2 = list(OrderedDict.fromkeys(item[0][1] for item in pmi_list))
        
        for (key1, key2), value in pmi_list:
            if key1 not in nested_dict:
                nested_dict[key1] = {}
            nested_dict[key1][key2] = value
        nested_dict['w1'] = w1
        nested_dict['w2'] = w2
        
        return nested_dict

def pmi_to_dict_adj_dict(pmi_dict):
    
    """
    Convert a flat PMI dictionary into a nested adjacency-like dictionary format.

    This function transforms a PMI dictionary of the form:
        {('w1', 'w2'): value}
    into a nested dictionary structure:
        {
            'w1': {'w2': value, ...},
            ...
            'w1': [...],  # list of unique first words
            'w2': [...]   # list of unique second words
        }

    Args:
        pmi_dict (dict): A dictionary where keys are tuples of word pairs (word1, word2),
                         and values are PMI scores (floats).

    Returns:
        dict: A nested dictionary where each first word maps to a dictionary of second words and their PMI scores,
              along with two additional keys:
                - 'w1': list of unique first words in the original PMI dictionary
                - 'w2': list of unique second words in the original PMI dictionary
    """
    nested_dict = {}
    w1 = list(OrderedDict.fromkeys(key[0] for key in pmi_dict.keys()))
    w2 = list(OrderedDict.fromkeys(key[1] for key in pmi_dict.keys()))

    for (key1, key2), value in pmi_dict.items():
        if key1 not in nested_dict:
            nested_dict[key1] = {}
        nested_dict[key1][key2] = value #[0]
    nested_dict['w1'] = w1
    nested_dict['w2'] = w2
    # print("w1: ", len(w1))
    # print("w2: ", len(w2))
    
    return nested_dict

def pmi_to_dict_adj(pmi_dict):
    """ Take a PMI dictionary in the format {('w1', 'w2'): value} and output a nested dictionary """
    nested_dict = {}

    # Use set comprehension to get unique keys for w1 and w2
    w1 = {key[0] for key in pmi_dict.keys()}
    w2 = {key[1] for key in pmi_dict.keys()}

    for (key1, key2), value in pmi_dict.items():
        if key1 not in nested_dict:
            nested_dict[key1] = {}
        nested_dict[key1][key2] = value #[0]

    nested_dict['w1'] = list(w1)
    nested_dict['w2'] = list(w2)

    return nested_dict

def dict2mat(dict_ini):
    # Extract unique words for rows (second words) and columns (first words)
    unique_words1 = sorted(set(pair[0] for pair in dict_ini.keys()))  # First word of bigram (columns)
    unique_words2 = sorted(set(pair[1] for pair in dict_ini.keys()))  # Second word of bigram (rows)
    
    # Map words to indices
    word1_to_index = {word: i for i, word in enumerate(unique_words1)}
    word2_to_index = {word: i for i, word in enumerate(unique_words2)}

    # Initialize the matrix with zeros
    n_rows = len(unique_words2)
    n_cols = len(unique_words1)
    matrix = np.zeros((n_rows, n_cols))

    # Populate the matrix
    for (word1, word2), value in dict_ini.items():
        row = word2_to_index[word2]
        col = word1_to_index[word1]
        matrix[row, col] = value

    # Convert the matrix to a DataFrame for better readability
    matrix_df = pd.DataFrame(matrix, index=unique_words2, columns=unique_words1)
    return matrix_df


def docs_distribution(baseSpace, tE):
    """
    Compute per-document and global term probability distributions from a set of texts.
    
    Vectorizes the Knowledge Base and toEval base, returning document-level and corpus-level distributions.
    
    Parameters:
        baseSpace (DataFrame): Knowledge Base texts, typically with a 'claims' column.
        tE (Series or list-like): Text examples to evaluate against the Knowledge Base.
    
    Returns:
        tuple: (Prob_KB_matrix, Corpus_dist, Count_matrix)
            - Prob_KB_matrix: Term probabilities for each Knowledge Base document.
            - Corpus_dist: Overall term probability distribution in the Knowledge Base.
            - Count_matrix: Term count matrix for all documents (Knowledge Base + tE).
    """
    # Combine baseSpace text with tE claims and handle NaN values
    KS_corpus = pd.concat([baseSpace, tE], axis=0).fillna("")

    # Vectorize the text to create the term-document matrix
    vectorizer = CountVectorizer()
    Count_matrix = vectorizer.fit_transform(KS_corpus)

    # Split term-document matrix into Knowledge Base and tE
    Old_matrix = Count_matrix[:len(baseSpace), :]  # Sparse matrix slicing

    # Compute term probability distributions for Knowledge Base documents
    row_sums = np.array(Old_matrix.sum(axis=1)).flatten()  # Ensure 1D array
    row_sums[row_sums == 0] = np.finfo(float).eps         # Replace zeros

    # Reshape row_sums to align with the sparse matrix's row structure
    row_sums_reshaped = row_sums[:, np.newaxis]  # Convert to column vector

    # Perform element-wise division safely
    Prob_KB_matrix = Old_matrix.multiply(1 / row_sums_reshaped)
    Prob_KB_matrix = Prob_KB_matrix.tocsr()  # Ensure CSR format

    # Compute overall term distribution in the Knowledge Base
    Count_overall = Old_matrix.sum(axis=0).A1  # Sum over all documents, convert to 1D array
    Corpus_dist = Count_overall / Count_overall.sum()

    return Prob_KB_matrix, Corpus_dist, Count_matrix


def new_distribution(Count_matrix, select_variation):
    """
    Compute updated corpus distribution and term distribution for a selected document subset.
    
    Returns the overall distribution for selected rows and the term distribution for the last row.
    
    Parameters:
        Count_matrix (csr_matrix): Sparse term-document count matrix.
        select_variation (list[int]): Indices of rows to include in the computation.
    
    Returns:
        tuple: (updated corpus distribution, term distribution for selected last row)
    """
    # Extract the rows corresponding to the selected variations
    New_Count_matrix = Count_matrix[select_variation, :]
    # Compute row sums (1D array)
    row_sums = New_Count_matrix.sum(axis=1).A1  # `.A1` converts sparse matrix result to 1D numpy array
    # Calculate row-wise probabilities (element-wise multiplication for sparse matrices)
    Variation_matrix = New_Count_matrix.multiply(1 / row_sums[:, None])

    Varations_dist = Variation_matrix.getrow(-1).toarray().flatten()  # Keep as sparse
    # Compute overall term counts (sum along columns)
    New_Count_overall = New_Count_matrix.sum(axis=0).A1  # `.A1` for 1D array
    # Compute overall term distribution
    updated_Corpus_dist = New_Count_overall / New_Count_overall.sum()

    return updated_Corpus_dist, Varations_dist
    

def combine_columns(data, selected_columns):
    """
    Combine text from specified DataFrame columns into a single column.
    
    Joins text row-wise from selected columns (or all if None) into one concatenated Series.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing text columns.
        selected_columns (list[str] or None): Column names to combine; all columns used if None.
    
    Returns:
        pd.Series: Combined text from the specified columns.
    """
    if selected_columns is None:
        # Use all columns if none are specified
        selected_columns = data.columns
    
    # Check if all selected columns exist in the DataFrame
    for col in selected_columns:
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")
    
    # Combine selected columns into a single column, row-wise
    combined_column = data[selected_columns].fillna('').agg(' '.join, axis=1)
    
    return combined_column




# =======================
# ANALYSIS UTILITIES
# =======================

# Score Analysis

def correl_labelScores(df):
    '''
    Computes Pearson and Spearman correlations between 'label' and all columns ending in score metrics (newness, uniqueness, difference, surprise)
    Parameters:
        df: pandas DataFrame with a 'label' column and several metric columns ending in 'ratio'
    Returns:
        A pandas DataFrame summarizing correlation coefficients and p-values for (newness, uniqueness, difference, surprise) metrics
    '''
    # Select columns ending with 'ratio'
    ratio_columns = [col for col in df.columns if col.endswith("ratio")]

    # Compute Pearson and Spearman correlations
    pearson_corr = df[["label"] + ratio_columns].corr(method="pearson")["label"].drop("label")
    spearman_corr = df[["label"] + ratio_columns].corr(method="spearman")["label"].drop("label")

    # Calculate p-values
    pearson_pvals = {col: pearsonr(df["label"], df[col])[1] for col in ratio_columns}
    spearman_pvals = {col: spearmanr(df["label"], df[col])[1] for col in ratio_columns}

    # Formatting the result as a DataFrame
    results = pd.DataFrame({
        0: ["Correlation label and metric", "pearson correlation (p-value)", "spearman correlation (p-value)"],
        1: ["newness", f"{pearson_corr['new_ratio']:.3f} ({pearson_pvals['new_ratio']:.3f})", f"{spearman_corr['new_ratio']:.3f} ({spearman_pvals['new_ratio']:.3f})"],
        2: ["uniqueness", f"{pearson_corr['uniq_ratio']:.3f} ({pearson_pvals['uniq_ratio']:.3f})", f"{spearman_corr['uniq_ratio']:.3f} ({spearman_pvals['uniq_ratio']:.3f})"],
        3: ["difference", f"{pearson_corr['diff_ratio']:.3f} ({pearson_pvals['diff_ratio']:.3f})", f"{spearman_corr['diff_ratio']:.3f} ({spearman_pvals['diff_ratio']:.3f})"],
        4: ["surprise divergence", f"{pearson_corr['surpDiv_ratio']:.3f} ({pearson_pvals['surpDiv_ratio']:.3f})", f"{spearman_corr['surpDiv_ratio']:.3f} ({spearman_pvals['surpDiv_ratio']:.3f})"]
    })

    return results


def ttest_metric(df):
    '''
    Performs Welch's t-tests between label groups (0 vs 1) for each metric ending in 'ratio'.
    Parameters:
        df: pandas DataFrame with a binary 'label' column and several 'ratio' metric columns
    Returns:
        A pandas DataFrame with t-statistics and p-values for each metric, formatted for display
    '''
    ratio_columns = [col for col in df.columns if col.endswith("ratio")]
    # Perform t-test for each ratio column
    t_test_results = {}
    for col in ratio_columns:
        group_0 = df[df["label"] == 0][col]
        group_1 = df[df["label"] == 1][col]

        # Perform t-test
        t_stat, p_value = ttest_ind(group_1, group_0, equal_var=False)  # Welch's t-test
        t_test_results[col] = {"t_stat": t_stat, "p_value": p_value}

    # Convert do dataframe cleaned up for visualization
    t_test_df = pd.DataFrame.from_dict(t_test_results, orient="index")
    t_test_df.insert(0, 'Metric', t_test_df.index)
    t_test_df.reset_index(drop=True, inplace=True)
    t_test_df.columns = [0,1,2]  # Rename the columns
    t_test_df.loc[-1] =  ["", 't_stat', 'p_value'] # Add a blank first row
    t_test_df.index = t_test_df.index + 1  # Shift index to make room for the new row
    t_test_df = t_test_df.sort_index()  # Re-sort the DataFrame by the index
    
    return t_test_df



def KTcorrel_metrics(df):
    '''
    Computes pairwise Kendall's Tau correlation between ratio metrics and formats results in a triangular matrix.
    Parameters:
        df: pandas DataFrame containing multiple scores (columns ending with "_ratio")
    Returns:
        A pandas DataFrame showing Kendall's Tau correlations (with p-values) in a lower triangular matrix format
    '''
    # Select only ratio columns
    ratio_columns = [col for col in df.columns if col.endswith("_ratio")]

    # Initialize an empty DataFrame with "-" for the upper triangle
    kendall_matrix = pd.DataFrame(np.full((len(ratio_columns), len(ratio_columns)), "-", dtype="object"))

    # Compute Kendall's Tau for the lower triangular part and diagonal
    for i, col1 in enumerate(ratio_columns):
        for j, col2 in enumerate(ratio_columns):
            if i >= j:  # Lower triangular and diagonal part
                tau, p_value = kendalltau(df[col1], df[col2])
                # Format the result as 'tau_value (p_value)'
                kendall_matrix.iloc[i, j] = f"{tau:.3f} ({p_value:.3f})"

    # Insert the ratio column names as the first column
    kendall_matrix.insert(0, "Metrics", ratio_columns)

    # Formatting for visualization
    kendall_matrix.loc[-1] = [""] + ratio_columns  # Add a blank row
    kendall_matrix.index = kendall_matrix.index + 1  # Shift the index
    kendall_matrix = kendall_matrix.sort_index()  # Sort the DataFrame to fix the index order
    kendall_matrix.columns = [0] + list(range(1, len(ratio_columns) + 1))  # First column = 0

    return kendall_matrix



def rbo_metrics(df, p=0.9):
    '''
    Computes Rank-Biased Overlap (RBO) between ranked lists of metrics (columns ending with "_ratio").
    Parameters:
        df: pandas DataFrame containing columns ending with "_ratio"
        p: persistence parameter for RBO, typically between 0.8 and 0.99. Defaulted to 0.9
    Returns:
        A pandas DataFrame showing pairwise RBO scores (formatted) in a lower triangular matrix
    '''
    # Select only ratio columns
    ratio_columns = [col for col in df.columns if col.endswith("_ratio")]

    # Initialize an empty DataFrame for the RBO matrix
    rbo_matrix = pd.DataFrame(np.ones((len(ratio_columns), len(ratio_columns))), columns=ratio_columns)
    rbo_matrix = rbo_matrix.astype("object")

    # Compute RBO similarity between each unique pair of ratio columns
    for col1, col2 in combinations(ratio_columns, 2):
        # Rank the values (descending)
        rank1 = df[col1].sort_values(ascending=False).index.tolist()
        rank2 = df[col2].sort_values(ascending=False).index.tolist()
        
        # Compute RBO (with given p value)
        rbo_score = rbo.RankingSimilarity(rank1, rank2).rbo(p=p)
        
        # Format the result
        rbo_matrix.loc[ratio_columns.index(col1), col2] = f"{rbo_score:.3f}"
        rbo_matrix.loc[ratio_columns.index(col2), col1] = f"{rbo_score:.3f}"

    # Fill the upper triangle with "-"
    for i in range(len(ratio_columns)):
        for j in range(i+1, len(ratio_columns)):
            rbo_matrix.iloc[i, j] = "-"

    # Format for visualization
    rbo_matrix.insert(0, "Metrics", ratio_columns)
    rbo_matrix.loc[-1] = [""] + ratio_columns  # Add a blank row
    rbo_matrix.index = rbo_matrix.index + 1  # Shift the index
    rbo_matrix = rbo_matrix.sort_index()  # Sort the DataFrame to fix the index order
    rbo_matrix.columns = [0] + list(range(1, len(ratio_columns) + 1))  # First column = 0

    return rbo_matrix




def rL_full(df):
    '''
    Fits a logistic regression model using four input metrics and Returns a custom summary table.

    Parameters:
        df: pandas DataFrame with the following columns:
            - 'label': accepted or rejected patents (0 or 1)
            - 'new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio': independent variables

    Returns:
        output_df: pandas DataFrame showing coefficients, standard errors, p-values,
                McFadden's Pseudo R-squared, and likelihood ratio test p-value
    '''
    # Define dependent and independent variables
    X = df[['new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio']]  # 4 ratios
    X = sm.add_constant(X)  # Add a constant for the intercept
    y = df['label']

    # Fit the logistic regression model
    model1 = sm.Logit(y, X)
    result1 = model1.fit()

    # Extract Coefficients, Standard Errors, P-values
    coefficients_model1 = result1.params
    std_err_model1 = result1.bse
    pvalues_model1 = result1.pvalues

    # Extract Pseudo R-squared and Likelihood Ratio (LLR) p-value
    r2_model1 = result1.prsquared  # McFadden's R-squared
    llr_p_value = result1.llr_pvalue

    # Prepare a DataFrame for the output
    output_df = pd.DataFrame({
        0: ["", 'const', 'Newness', 'Difference', 'Uniqueness', 'Surprise', 'Pseudo R-square', 'LLR p-value'],
        1: [
            'coef',
            round(coefficients_model1.get('const', np.nan), 3),
            round(coefficients_model1.get('new_ratio', np.nan), 3),
            round(coefficients_model1.get('diff_ratio', np.nan), 3),
            round(coefficients_model1.get('uniq_ratio', np.nan), 3),
            round(coefficients_model1.get('surpDiv_ratio', np.nan), 3),
            round(r2_model1, 3),  # Pseudo R-square
            round(llr_p_value, 3)  # LLR p-value
        ],
        2: [
            'std err',
            round(std_err_model1.get('const', np.nan), 3),
            round(std_err_model1.get('new_ratio', np.nan), 3),
            round(std_err_model1.get('diff_ratio', np.nan), 3),
            round(std_err_model1.get('uniq_ratio', np.nan), 3),
            round(std_err_model1.get('surpDiv_ratio', np.nan), 3),
            np.nan,  # No std err for Pseudo R-square
            np.nan   # No std err for LLR p-value
        ],
        3: [
            'P>|t|',
            round(pvalues_model1.get('const', np.nan), 3),
            round(pvalues_model1.get('new_ratio', np.nan), 3),
            round(pvalues_model1.get('diff_ratio', np.nan), 3),
            round(pvalues_model1.get('uniq_ratio', np.nan), 3),
            round(pvalues_model1.get('surpDiv_ratio', np.nan), 3),
            np.nan,  # No p-value for Pseudo R-square
            np.nan   # No p-value for LLR p-value
        ]
    })


    return output_df




def rL_full_aVC(df, base):
    '''
    Fit a logistic regression model to predict a binary label based on metrics AND control variables 
        vocabRel: nb of vocab of patent/full nb of vocab in base
        wordsRel: nb of vocab of patent/full nb of words in base
        lexicDiv: nb of vocab of patent/full nb of words in base
        rank_mean: rank of word with average frequency in the patent
        rank_meanPlusStd: rank of word with average + 1 standard deviation frequency in the patent 
        (rank_meanMinusStd excluded for collinearity reason - almost 100% corrleated with wordsRel)


    Parameters:
        df   : pandas DataFrame with required columns including control varialbles, scores (ending with "_ration") and a 'label' column.
        base : "ES" or "KS" for Expectation or knowledge base/space.

    Returns:
        output_df : pandas DataFrame summarizing coefficients, standard errors, p-values,
                    and model metrics (Pseudo R² and LLR p-value)
    '''
    required_columns = [
        'new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio',
        f'vocabRel_{base}', f'wordsRel_{base}', 'lexicDiv', 
        'rank_mean', 'rank_meanPlusStd']   

    # Ensure all required columns are present
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the dataframe: {missing}")

    # Define dependent and independent variables
    X = df[['new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio', f'vocabRel_{base}', f'wordsRel_{base}', 'lexicDiv',
            'rank_mean', 'rank_meanPlusStd']]  # , 'rank_meanMinusStd']]  
    X = sm.add_constant(X)  # Add a constant for the intercept
    y = df['label']

    # Fit the logistic regression model
    model1 = sm.Logit(y, X)
    result1 = model1.fit()

    # Extract Coefficients, Standard Errors, P-values
    coefficients_model1 = result1.params
    std_err_model1 = result1.bse
    pvalues_model1 = result1.pvalues

    # Extract Pseudo R-squared and Likelihood Ratio (LLR) p-value
    r2_model1 = result1.prsquared  # McFadden's R-squared
    llr_p_value = result1.llr_pvalue

    # Prepare a DataFrame for the output
    output_df = pd.DataFrame({
        '0': ["", 'const', 'new_ratio', 'diff_ratio', 'uniq_ratio', 'surpDiv_ratio', 
                   f'vocabRel_{base}', f'wordsRel_{base}', 'lexicDiv', 
                   'rank_mean', 'rank_meanPlusStd', 'Pseudo R-square', 'LLR p-value'],
        '1': [
            'coef', 
            round(coefficients_model1.get('const', np.nan), 3),
            round(coefficients_model1.get('new_ratio', np.nan), 3),
            round(coefficients_model1.get('diff_ratio', np.nan), 3),
            round(coefficients_model1.get('uniq_ratio', np.nan), 3),
            round(coefficients_model1.get('surpDiv_ratio', np.nan), 3),
            round(coefficients_model1.get(f'vocabRel_{base}', np.nan), 3),
            round(coefficients_model1.get(f'wordsRel_{base}', np.nan), 3),
            round(coefficients_model1.get('lexicDiv', np.nan), 3),
            round(coefficients_model1.get('rank_mean', np.nan), 3),
            round(coefficients_model1.get('rank_meanPlusStd', np.nan), 3),
            round(r2_model1, 3),  # Pseudo R-square
            round(llr_p_value, 3)  # LLR p-value
        ],
        '2': [
            'std err', 
            round(std_err_model1.get('const', np.nan), 3),
            round(std_err_model1.get('new_ratio', np.nan), 3),
            round(std_err_model1.get('diff_ratio', np.nan), 3),
            round(std_err_model1.get('uniq_ratio', np.nan), 3),
            round(std_err_model1.get('surpDiv_ratio', np.nan), 3),
            round(std_err_model1.get(f'vocabRel_{base}', np.nan), 3),
            round(std_err_model1.get(f'wordsRel_{base}', np.nan), 3),
            round(std_err_model1.get('lexicDiv', np.nan), 3),
            round(std_err_model1.get('rank_mean', np.nan), 3),
            round(std_err_model1.get('rank_meanPlusStd', np.nan), 3),
            np.nan,  # No std err for Pseudo R-square
            np.nan   # No std err for LLR p-value
        ],
        '3': [
            'P>|t|', 
            round(pvalues_model1.get('const', np.nan), 3),
            round(pvalues_model1.get('new_ratio', np.nan), 3),
            round(pvalues_model1.get('diff_ratio', np.nan), 3),
            round(pvalues_model1.get('uniq_ratio', np.nan), 3),
            round(pvalues_model1.get('surpDiv_ratio', np.nan), 3),
            round(pvalues_model1.get(f'vocabRel_{base}', np.nan), 3),
            round(pvalues_model1.get(f'wordsRel_{base}', np.nan), 3),
            round(pvalues_model1.get('lexicDiv', np.nan), 3),
            round(pvalues_model1.get('rank_mean', np.nan), 3),
            round(pvalues_model1.get('rank_meanPlusStd', np.nan), 3),
            np.nan,  # No p-value for Pseudo R-square
            np.nan   # No p-value for LLR p-value
        ]
    })

    return output_df




def rL_metricSeparate(df):
    '''
    Performs separate univariate logistic regressions for each of several metrics and summarizes the results in a formatted table.

    Parameters:
    - df: A DataFrame containing a binary `label` column and four metric columns: `new_ratio`, `uniq_ratio`, `diff_ratio`, and `surpDiv_ratio`.

    Returns:
    - final_df: A DataFrame summarizing the coefficients, standard errors, p-values, pseudo R², and LLR p-values for each univariate logistic regression model.
    '''
    # Define dependent variable
    y = df['label']

    # List of metrics (independent variables)
    metrics = ['new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio']

    # Create an empty list to store DataFrames
    df_list = []

    for metric in metrics:
        # Define independent variable (X) as the metric
        X = df[[metric]]
        X = sm.add_constant(X)  # Add a constant for the intercept

        # Fit the logistic regression model
        model = sm.Logit(y, X)
        result = model.fit()

        # Extract coefficients, standard errors, p-values
        coef_const = round(result.params['const'], 3)
        coef_metric = round(result.params[metric], 3)
        std_err_const = round(result.bse['const'], 3)
        std_err_metric = round(result.bse[metric], 3)
        p_value_const = round(result.pvalues['const'], 3)
        p_value_metric = round(result.pvalues[metric], 3)

        # Extract Pseudo R-squared and Likelihood Ratio (LLR) p-value
        r2 = round(result.prsquared, 3)
        llr_p_value = f"{result.llr_pvalue:.3E}"

        # Create a DataFrame for this metric
        metric_df = pd.DataFrame({
            0: ['', 'const', metric, 'Pseudo R-square', 'LLR p-value'],
            1: ['coef', coef_const, coef_metric, r2, llr_p_value],
            2: ['std err', std_err_const, std_err_metric, "", ""],
            3: ['P>|t|', p_value_const, p_value_metric, "", ""]
        })

        # Append DataFrame and a blank row
        df_list.append(metric_df)
        df_list.append(pd.DataFrame({0: [""], 1: [""], 2: [""], 3: [""]}))  # Blank row

    # Concatenate all DataFrames
    final_df = pd.concat(df_list, ignore_index=True)

    return final_df


def rL_metricSeparate_aVC(df):
    '''
    Fit a logistic regression model for different metrics AND control variables and return a DataFrame with coefficients, standard errors, p-values, and model statistics.
    
    Parameters:
        df: DataFrame containing the data, including the dependent variable 'label' and various metrics as control variables:
            Control variables are:
                vocabRel: nb of vocab of patent/full nb of vocab in base
                wordsRel: nb of vocab of patent/full nb of words in base
                lexicDiv: nb of vocab of patent/full nb of words in base
                rank_mean: rank of word with average frequency in the patent
                rank_meanPlusStd: rank of word with average + 1 standard deviation frequency in the patent 
                (rank_meanMinusStd excluded for collinearity reason - almost 100% corrleated with wordsRel)
        
    Returns:
        final_df: DataFrame containing the regression results (coefficients, standard errors, p-values, Pseudo R-squared, LLR p-value) for each metric.
    '''
    # Define dependent variable
    y = df['label']

    # List of metrics (independent variables)
    metrics = ['new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio']

    # Create an empty list to store DataFrames
    df_list = []

    # Loop through each metric and refit the model
    for metric in metrics:
        base = "KS"
        if metric == 'surpDiv_ratio':
            base = "ES"
            
        # Define independent variable (X) as the metric
        X = df[[metric, f'vocabRel_{base}', f'wordsRel_{base}', 'lexicDiv', 
                   'rank_mean', 'rank_meanPlusStd']]
        X = sm.add_constant(X)  # Add a constant for the intercept

        # Fit the logistic regression model
        model = sm.Logit(y, X)
        result = model.fit()

        # Extract coefficients, standard errors, p-values
        coef_const = round(result.params['const'], 3)
        coef_metric = round(result.params[metric], 3)
        coef_vocabRel = round(result.params[f'vocabRel_{base}'], 3)
        coef_wordsRel = round(result.params[f'wordsRel_{base}'], 3)
        coef_lexicDiv = round(result.params['lexicDiv'], 3)
        coef_rank_mean = round(result.params['rank_mean'], 3)
        coef_rank_meanPlusStd = round(result.params['rank_meanPlusStd'], 3)

        std_err_const = round(result.bse['const'], 3)
        std_err_metric = round(result.bse[metric], 3)
        std_err_vocabRel = round(result.bse[f'vocabRel_{base}'], 3)
        std_err_wordsRel = round(result.bse[f'wordsRel_{base}'], 3)
        std_err_lexicDiv = round(result.bse['lexicDiv'], 3)
        std_err_rank_mean = round(result.bse['rank_mean'], 3)
        std_err_rank_meanPlusStd = round(result.bse['rank_meanPlusStd'], 3)

        p_value_const = round(result.pvalues['const'], 3)
        p_value_metric = round(result.pvalues[metric], 3)
        p_value_vocabRel = round(result.pvalues[f'vocabRel_{base}'], 3)
        p_value_wordsRel = round(result.pvalues[f'wordsRel_{base}'], 3)
        p_value_lexicDiv = round(result.pvalues['lexicDiv'], 3)
        p_value_rank_mean = round(result.pvalues['rank_mean'], 3)
        p_value_rank_meanPlusStd = round(result.pvalues['rank_meanPlusStd'], 3)

        # Extract Pseudo R-squared and Likelihood Ratio (LLR) p-value
        r2 = round(result.prsquared, 3)
        llr_p_value = f"{result.llr_pvalue:.3E}"

        # Create a DataFrame for this metric
        metric_df = pd.DataFrame({
            0: ['', 'const', metric, f'vocabRel_{base}', f'wordsRel_{base}', 'lexicDiv', 'rank_mean', 'rank_meanPlusStd', 'Pseudo R-square', 'LLR p-value'],
            1: ['coef', coef_const, coef_metric, coef_vocabRel, coef_wordsRel, coef_lexicDiv, coef_rank_mean, coef_rank_meanPlusStd, r2, llr_p_value],
            2: ['std err', std_err_const, std_err_metric, std_err_vocabRel, std_err_wordsRel, std_err_lexicDiv, std_err_rank_mean, std_err_rank_meanPlusStd,  "", ""],
            3: ['P>|t|', p_value_const, p_value_metric, p_value_vocabRel, p_value_wordsRel, p_value_lexicDiv, p_value_rank_mean, p_value_rank_meanPlusStd,  "", ""]
        })

        # Append DataFrame and a blank row
        df_list.append(metric_df)
        df_list.append(pd.DataFrame({0: [""], 1: [""], 2: [""], 3: [""]}))  # Blank row

    # Concatenate all DataFrames
    final_df = pd.concat(df_list, ignore_index=True)

    return final_df

##### Control Variables additions #####


def count_words(df, columns):
    """
    Count the total number of words across multiple string columns in a DataFrame row-wise. Used to count words in 1 patent, in the selected columns

    Parameters:
        df (pd.DataFrame): The DataFrame containing text columns.
        columns (list of str): The list of column names to include in the word count.

    Returns:
        pd.Series: A Series with the total word count per row.
    """
    return df[columns].apply(lambda col: col.str.split().str.len()).sum(axis=1)

# Function to count vocabulary size (unique words) in selected columns
def count_vocab(df, columns):
    """
    Count the number of unique words across multiple string columns in a DataFrame row-wise.

    Parameters:
        df (pd.DataFrame): The DataFrame containing text columns.
        columns (list of str): The list of column names to include in the vocabulary count.

    Returns:
        pd.Series: A Series with the count of unique words per row.
    """
    # Combine the text from all selected columns for each row
    combined_text = df[columns].fillna('').apply(lambda row: ' '.join(row.astype(str)), axis=1) #.fillna('')
    
    # Split the combined text into words, convert to a set (unique words), and count the length of the set
    unique_word_counts = combined_text.apply(lambda text: len(set(text.split())))
    
    # Return the total unique word counts for each row
    return unique_word_counts

def count_total_vocab(df, columns):
    '''
    Count the total vocabulary size (unique words) across all rows and columns.
    Parameters:
        df (pandas DataFrame): DataFrame containing text data.
        columns (list of str): List of column names from df to include in the vocabulary count.
    Returns:
        int: Total number of unique words across the specified columns in all rows.
    '''
    word_counter = Counter()
    for _, row in df[columns].astype(str).iterrows():
        row = row.fillna('') # added after updating code
        words = ' '.join(row).split()  # Tokenize row text
        word_counter.update(words)  # Update word count
    
    return len(word_counter)  # Return unique word count


def rank_words(df, columns, i):
    '''
    Rank words based on their frequency across selected columns and calculate specific word ranking indices.
    Parameters:
        df (pandas DataFrame): DataFrame containing text data.
        columns (list of str): List of column names from df to include in word ranking.
        i (int): Row index for which the word frequency ranking will be calculated.
    Returns:
        tuple: Indices for mean word frequency, mean + standard deviation, and mean - standard deviation.
    '''
    # Combine the text from all selected columns for each row
    combined_text = df[i:i+1][columns].apply(lambda row: ' '.join(row.astype(str)), axis=1)
    
    # Tokenize the combined text into words and flatten the list of words
    all_words = combined_text.str.split().sum()
    
    # Count the frequency of each word
    word_counts = Counter(all_words)
    
    # Convert to a DataFrame and sort by frequency (highest first)
    word_freq_df = pd.DataFrame(word_counts.items(), columns=["Word", "Frequency"])
    word_rank_df = word_freq_df.sort_values(by="Frequency", ascending=False).reset_index(drop=True)
    
    mean_idx = 0
    meanPlusStd_idx = 0
    meanMinusStd_idx = 0

    if len(word_rank_df)>0:
        mean_idx = word_rank_df[word_rank_df["Frequency"] <= np.mean(word_rank_df["Frequency"])].index[0]
        threshold = np.mean(word_rank_df["Frequency"]) + np.std(word_rank_df["Frequency"])
        meanPlusStd_idx = word_rank_df[word_rank_df["Frequency"] <= np.mean(word_rank_df["Frequency"]) + np.std(word_rank_df["Frequency"])].index[0]
        threshold = np.mean(word_rank_df["Frequency"]) - np.std(word_rank_df["Frequency"])
        if threshold <  word_rank_df.iloc[-1]["Frequency"]:
            meanMinusStd_idx = len(word_rank_df)
        else:
            meanMinusStd_idx = word_rank_df[word_rank_df["Frequency"] <= threshold].index[0]
    
    return mean_idx, meanPlusStd_idx, meanMinusStd_idx

def addControlVars(ipcList, yearList, sC, pathData, pathMetrics):
    '''
    Add control variables (word counts, vocabulary sizes, rankings, relations) to a DataFrame
    for a given list of IPC, years, and selected columns, then merge it with metrics data.
    Parameters:
        ipcList (list): List of IPC identifiers to process.
        yearList (list): List of years to process.
        sC (list of lists): List of selected column names
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab).
        pathMetrics (str): Path to the metrics directory (containing metrics)
    Returns:
        None (Saves processed DataFrames to CSV files).
    '''

    pathClean = os.path.join(pathData, "csv_clean")
    pathMetricsRaw = os.path.join(pathMetrics, "metrics_raw")
    pathOutput = os.path.join(pathMetrics, "metrics")

    for ipc in ipcList:
        print(f"{ipc}")
        for year in yearList:
            print(f"    {year}")
            for selectedColumns in sC:
                print(f"         {selectedColumns}")
                df_tE = pd.read_csv(os.path.join(pathClean, "tE", f"{year}_{ipc}_tE_cleaned.csv"))
                df_KS = pd.read_csv(os.path.join(pathClean, "KS", f"{year}_{ipc}_KS_cleaned.csv"))
                

                # Create dynamic suffix based on the selected columns
                suffix = "".join([col[0].lower() for col in selectedColumns])
    
                # Apply the dynamic suffix to new columns
                df_tE[f"tE_words"] = count_words(df_tE, selectedColumns)
                df_tE[f"tE_vocab"] = count_vocab(df_tE, selectedColumns)
                df_tE[f"KS_words"] = np.sum(count_words(df_KS, selectedColumns))
                df_tE[f"KS_vocab"] = count_total_vocab(df_KS, selectedColumns)
                df_KS = None # free memory

                df_ES = pd.read_csv(os.path.join(pathClean, "ES", f"{year}_{ipc}_ES_cleaned.csv"))
                df_tE[f"ES_words"] = np.sum(count_words(df_ES, selectedColumns))
                df_tE[f"ES_vocab"] = count_total_vocab(df_ES, selectedColumns)
                df_ES = None # free memory

                # Apply the dynamic suffix to the 'rank' columns as well
                rank_columns = [f"rank_mean", f"rank_meanPlusStd", f"rank_meanMinusStd"]
                df_tE[rank_columns] = df_tE.apply(
                    lambda row: pd.Series(rank_words(df_tE, selectedColumns, row.name)), axis=1
                )
    
                # Add dynamic suffix to relation columns
                df_tE[f"vocabRel_KS"] = df_tE[f"tE_vocab"] / df_tE[f"KS_vocab"]
                df_tE[f"vocabRel_ES"] = df_tE[f"tE_vocab"] / df_tE[f"ES_vocab"]
    
                df_tE[f"wordsRel_KS"] = df_tE[f"tE_words"] / df_tE[f"KS_words"]
                df_tE[f"wordsRel_ES"] = df_tE[f"tE_words"] / df_tE[f"ES_words"]
    
                df_tE[f"lexicDiv"] = df_tE[f"tE_vocab"] / df_tE[f"tE_words"]
    
    
                columns_to_keep = ["application_number", "label"] + df_tE.columns[df_tE.columns.get_loc(f"tE_words"):].tolist()
                df_tE=df_tE[columns_to_keep]
                file_metrics = os.listdir(pathMetricsRaw)
                
                # Loop through the file names and find the one that matches
                selected_file = None
                for file in file_metrics:
                    if ipc in file and str(year) in file:
                        # Check if filename ends with '_Metrics.csv'
                        if not file.endswith("_Metrics.csv"):
                            continue
                        
                        vs_part = file.split("vs_")[1].split("_Metrics")[0]  # Extract part after 'vs_' and before '_Metrics'
                        vs_columns = vs_part.split("_") 
                        
                        # Check if all selected columns are present in the vs_part
                        if all(col in vs_columns for col in selectedColumns):
                            selected_file = file
                            break
                
                if selected_file:
                    print(f"Selected file: {selected_file}")
                else:
                    print("No matching file found.")
    
                
                metrics_df = pd.read_csv(os.path.join(pathMetricsRaw, selected_file))
    
                df = pd.merge(metrics_df, df_tE, on=["application_number", "label"])
                new_file = selected_file.replace('.csv', '_aVC.csv')
                df.to_csv(os.path.join(pathOutput, new_file))



##### CD-Index #####

def applicationPatent_match(listYear, pathData):
    """
    Extracts patent application metadata from JSON files by year and exports them as CSV files.

    This function reads JSON files containing patent citation data for a given range of years,
    extracts relevant fields (application number, patent number, publication number, main IPC label, and decision),
    and saves one CSV file per year containing this data.

    Parameters
    ----------
    listYear : list or range
        A list or range of years (e.g., range(2012, 2017)) for which the JSON data will be processed.

    pathData : str
        The base path where the data is stored. It should contain a subdirectory "jsonData" with one folder per year.

    Returns
    -------
    None
        The function writes one CSV file per year to the "app_pat_match" subdirectory of `pathData`.
        No value is returned.
    
    Notes
    -----
    - Requires each JSON file to contain the following keys:
      'application_number', 'patent_number', 'publication_number', 'main_ipcr_label', 'decision'.
    - Files with invalid or empty JSON content are skipped.
    - Outputs are saved as CSVs in: `pathData/app_pat_match/app_pat_match_<year>.csv`.
    """
    pathJson = os.path.join(pathData, "jsonData")

    citations_Year = {ipc: {
        'application_number': [], 'patent_number': [], 'publication_number': [], 'main_ipc': [], 'decision': []
    } for ipc in listYear}

    for year in (listYear):
        pathYear = pathJson + f"/{year}/"  # Updates with variable year
        jsonNamesYear = [f for f in os.listdir(pathYear) if os.path.isfile(os.path.join(pathYear, f))] 

        # Total number of JSON files
        total_files = len(jsonNamesYear)

        # Creates a list of patents for each IPC class with batch-size tqdm
        for j in tqdm(range(0, total_files)):
            patent_path = pathYear + jsonNamesYear[j]
            if os.path.exists(patent_path) and os.path.getsize(patent_path) > 0:
                with open(patent_path) as f:
                    try:
                        d = json.load(f)  # Load JSON into d
                    except json.JSONDecodeError:
                        continue
            else:
                print("File does not exist or is empty:", patent_path)
        
            # Creating the lists for the other information
            citations_Year[year]['application_number'].append(d['application_number'])
            citations_Year[year]['patent_number'].append(d['patent_number'])
            citations_Year[year]['publication_number'].append(d['publication_number'])
            citations_Year[year]['main_ipc'].append(d['main_ipcr_label'])
            citations_Year[year]['decision'].append(d['decision'])


    path_application_patent_match = os.path.join(pathData, "app_pat_match")

    for year, data in citations_Year.items():
        df = pd.DataFrame(data)
        df.insert(0, 'year', year)  # Insert 'year' as the first column
        filename = os.path.join(path_application_patent_match, f"app_pat_match_{year}.csv")
        df.to_csv(filename, index=False)


def createGraph(pathData):
    '''
    Creates a directed citation graph of U.S. utility patents and returns the graph with valid patent IDs.

    Parameters:
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab, citationsData).

    Returns:
        graph (cdindex.Graph): A graph object containing patents as nodes and citations as directed edges.
        valid_patent_ids (set): A set of valid patent IDs that are used as nodes in the graph.
    '''

    pathPatent = os.path.join(pathData, "citationsData","g_patent.tsv")
    pathGranted = os.path.join(pathData, "citationsData", "g_us_patent_citation.tsv")

    # Load the TSV file into a DataFrame
    patent_df = pd.read_csv(pathPatent, sep='\t', on_bad_lines='skip')

    # Clean and filter
    patent_df = patent_df[patent_df["patent_type"] == "utility"]
    patent_df = patent_df[patent_df["patent_id"].notna()]
    patent_df["patent_id"] = patent_df["patent_id"].astype(str)
    patent_df = patent_df[patent_df["patent_id"].str.isnumeric()]
    patent_df["patent_date"] = pd.to_datetime(patent_df["patent_date"]).dt.tz_localize("UTC")

    valid_patent_ids = set(patent_df["patent_id"])

    # Use a vectorized approach to create the dictionary list
    pyvertices = patent_df[['patent_id', 'patent_date']].rename(columns={'patent_id': 'name', 'patent_date': 'time'}).to_dict(orient="records")
    graph = cdindex.Graph()

    # add vertices
    for vertex in tqdm(pyvertices, mininterval=1.0):
        graph.add_vertex(vertex["name"], int(datetime.datetime.timestamp(vertex["time"])))

    # for memory
    del pyvertices
    valid_patent_ids = set(patent_df["patent_id"])
    del patent_df

    # From granted files path
    chunksize = 10_000_000  # Read 10 million rows at a time
    total_rows = 0  # Counter to track progress
    grant_file= pathGranted

    for chunk in pd.read_csv(grant_file, sep='\t', usecols=["patent_id", "citation_patent_id"], chunksize=chunksize, on_bad_lines='skip'):
        print(f"Processing rows {total_rows} to {total_rows + len(chunk)}")
        
        # Convert to string if needed
        chunk["patent_id"] = chunk["patent_id"].astype(str)
        chunk["citation_patent_id"] = chunk["citation_patent_id"].astype(str)

        filtered_df = chunk[chunk["patent_id"].isin(valid_patent_ids) & chunk["citation_patent_id"].isin(valid_patent_ids)]
        
        pyedges = filtered_df.rename(columns={'patent_id': 'source', 'citation_patent_id': 'target'}) \
                .astype(str) \
                .to_dict(orient='records')
        
        # add edges
        for edge in tqdm(pyedges, mininterval=1.0):
            graph.add_edge(edge["source"], edge["target"])


        total_rows += len(chunk)
        
    print(f"Finished processing {total_rows} rows.")
    del chunk
    del filtered_df
    del pyedges

    graph.prepare_for_searching()

    return graph, valid_patent_ids


def calculateCDI(graph, valid_patent_ids, pathMetrics):
    '''
    Calculates the CD5 index for a set of valid patents over a 5-year citation window.

    Parameters:
        graph (cdindex.Graph): A citation graph with vertices and edges representing patents and citations.
        valid_patent_ids (set): A set of patent IDs for which to compute the CD5 index.
        pathMetrics (str): Path to the directory containing the metrics files.

    Returns:
        None. Writes a DataFrame with Patent IDs and their CD5 index to disk.
    '''
    pathCD = os.path.join(pathMetrics, "cd_index", "cd_index_5_results.csv")

    window = int(datetime.timedelta(days=1825).total_seconds())

    results = [
        (pid, graph.cdindex(pid, window))
        for pid in valid_patent_ids
    ]

    df_cd5 = pd.DataFrame(results, columns=["Patent ID", "CD5"])

    df_cd5.to_csv(pathCD, index=False)



def createCDindex(pathCDI, path_patent, df_CDI=pd.DataFrame(), yearList=range(2012, 2017)):
    '''
    Merge CD index data with patent application numbers across all patents (2012-2016). Application numbers were kept as ID when importing data, but patent numbers for CD-Index. This function creates the link.
    
    Parameters:
        pathCDI: Path to the CSV file containing CD index data, including patent id and CD-Index.
        path_patent: Path pattern to yearly patent data CSVs with a placeholder (at least) columns year, application_number, patent_number,	publication_number,	main_ipc, decision.
        df_CDI: Optional preloaded DataFrame for CD index data. If not provided, it will be read from pathCDI.
        yearList : List of years to compute 
    
    Returns:
        final_df: DataFrame with columns ['Patent ID', 'application_number', 'CD5'] after merging.
    '''
    if df_CDI.empty:
        df_CDI = pd.read_csv(pathCDI)
    
    # Load all years (2012-2016) into a single DataFrame
    df_link_list = [pd.read_csv(path_patent.format(year)) for year in yearList]
    df_link_combined = pd.concat(df_link_list, ignore_index=True)

    # Ensure both columns are of the same type
    df_CDI['Patent ID'] = df_CDI['Patent ID'].astype(str)
    df_link_combined['patent_number'] = df_link_combined['patent_number'].apply(lambda x: str(int(float(x))) if str(x).replace('.', '', 1).isdigit() else str(x))

    # Merge df_CDI with the combined df_link
    final_df = df_CDI.merge(df_link_combined, left_on='Patent ID', right_on='patent_number', how='inner')

    # Keep only relevant columns
    final_df = final_df[['Patent ID', 'application_number', 'CD5']]
    
    return final_df


def cdI_correl(df, df_cd_final):
    '''
    Compute Pearson correlations between CD5 and several patent text metrics keeping only 

    Parameters:
        df: DataFrame with patent-level metrics and a 'label' column.
        df_cd_final: DataFrame with 'application_number' and 'CD5' values.

    Output:
        DataFrame with correlation, p-value, means, and standard deviations 
        of metrics and CD5, including % of accepted matched entries.
    '''
    # Filter out rejected patents - no CD-index - out because merging does this anyway
    # df_filtered = df[df['label'] == 1]

    merged_df = pd.merge(df, df_cd_final, left_on='application_number', right_on='application_number', how='inner')

    # Define the variables for correlation
    target_col = 'CD5'
    feature_cols = ['new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio']

    # Compute correlation and p-value
    results = []
    for col in feature_cols:
        valid_data = merged_df[[target_col, col]].dropna()  # Remove NaN values

        # Ensure enough unique values before computing correlation
        if len(valid_data) > 1 and valid_data[target_col].nunique() > 1 and valid_data[col].nunique() > 1:
            corr, pval = pearsonr(valid_data[target_col], valid_data[col])
        else:
            corr, pval = float('nan'), float('nan')
        mean_cdI = valid_data["CD5"].mean()
        std_cdI= valid_data["CD5"].std()

        results.append([col, corr, pval, mean_cdI, std_cdI])
    
    # Create DataFrame with correlation and p-values
    correlations_cleaned = pd.DataFrame(results, columns=['Metric', 'Correlation_with_CD5', 'p-value', "Mean CD-index", "std CD-index"])

    metric_names = {
        'new_ratio': 'Newness',
        'uniq_ratio': 'Unique',
        'diff_ratio': 'Difference',
        'surpDiv_ratio': 'Surprise'
    }
    correlations_cleaned['Metric'] = correlations_cleaned['Metric'].map(metric_names)

    ratio_cols = [col for col in df.columns if col.endswith("ratio")]

    correlations_cleaned['Mean'] = df[ratio_cols].mean().values
    correlations_cleaned['Std'] = df[ratio_cols].std().values

    # Compute "% of accepted matched"
    matched_percentage = len(merged_df.dropna(subset=[target_col] + feature_cols)) / len(merged_df) * 100

    correlations_cleaned = pd.concat([
        correlations_cleaned,
        pd.DataFrame({'Metric': ['% of accepted matched'], 'Correlation_with_CD5': [matched_percentage], 'p-value': [float('nan')]})
    ], ignore_index=True)

    # Ensure correct order
    metric_order = ["Newness", "Unique", "Difference", "Surprise", "% of accepted matched"]
    correlations_cleaned['Metric'] = pd.Categorical(correlations_cleaned['Metric'], categories=metric_order, ordered=True)
    correlations_cleaned = correlations_cleaned.sort_values("Metric").reset_index(drop=True)

    # Round values to 3 decimals
    correlations_cleaned['Correlation_with_CD5'] = correlations_cleaned['Correlation_with_CD5'].round(3)
    correlations_cleaned['p-value'] = correlations_cleaned['p-value'].round(3)

    correlations_cleaned.columns = [0, 1, 2, 3, 4, 5, 6]

    return(correlations_cleaned)




##### Concatenate dfs ######

def merge_dataframes_with_blank_lines(df_list, df_names):
    """
    Merges a list of DataFrames into a single DataFrame with blank rows between them.
    Ensures column alignment and fills missing values with blanks instead of NaN.
    Adds DataFrame names in the first column of blank rows and avoids an extra blank row.

    Parameters:
        df_list (list of pd.DataFrame): List of DataFrames to merge.
        df_names (list of str): List of names corresponding to the DataFrames in df_list.

    Returns:
        Merged DataFrame with blank rows in between and names in the first column.
    """
    # Collect all column names and ensure they are strings
    all_columns = set()
    for df in df_list:
        all_columns.update(map(str, df.columns))  
    all_columns = sorted(all_columns, key=str)  

    # Standardize DataFrames by including all columns
    standardized_dfs = [df.rename(columns=str).reindex(columns=all_columns, fill_value="") for df in df_list]

    # Create a blank row DataFrame with the correct columns
    blank_row = pd.DataFrame([[""] * len(all_columns)], columns=all_columns)

    # Create a list to hold the DataFrames with their names
    merged_dfs_with_names = []

    # Interleave blank rows between DataFrames and add the names in the first column of the blank row
    for df, name in zip(standardized_dfs, df_names):
        # Create a blank row with the name of the DataFrame in the first column and blanks for others
        name_row = pd.DataFrame([[name] + [""] * (len(all_columns)-1)], columns=all_columns)
        merged_dfs_with_names.append(name_row)  # Add the name row
        merged_dfs_with_names.append(df)  # Add the DataFrame itself
        merged_dfs_with_names.append(blank_row)  # Add a blank row after each DataFrame

    # Merge all DataFrames, interleaving blank rows between them
    merged_df = pd.concat(merged_dfs_with_names, ignore_index=True)

    return merged_df

def output_to_excel(df_list, sheet_names, pathMetrics):
    """
    Writes multiple DataFrames to an Excel file with one sheet per DataFrame.
    Parameters:
        df_list: (list of pd.DataFrame): List of DataFrames to write.
        sheet_names: (list of str): Corresponding sheet names (max 31 chars each).
        pathMetrics (str): Path to the metrics directory (containing metrics)
    """

    output_file = os.path.join(pathMetrics, "metricAnalysis", "metricAnalysis.xlsx")

    try:
        with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
            for df, sheet_name in zip(df_list, sheet_names):
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"DataFrames written to {output_file}")
    except Exception as e:
        print(f"Error writing to Excel: {e}")


def metricAnalysis(pathMetrics="C:/Users/edgar/OneDrive/Bureau/Ecole/HEC/A24/BrevetNLP/exemple données/metrics/à traiter/",
                   pathData=None,
                   p=0.9,
                   yearList=range(2012, 2017)):
    """
    Computes all ranking metric DataFrames for each file in the path and returns them with sheet names.

    Parameters:
        pathMetrics (str): Path to the metrics directory (containing metrics)
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab).
        p (float): RBO parameter (usually between 0.8 and 0.98)         Default : 0.9
        yearList: List of years to compute the analysis for

    Returns:
        df_list (list of pd.DataFrame): List of DataFrames for Excel output.
        sheet_names (list of str): Corresponding Excel sheet names (max 31 chars).
    """
    # Initialize a list to store DataFrames and their corresponding sheet names
    df_list = []
    sheet_names = []

    pathCDI = os.path.join(pathMetrics, "cd_index", "cd_index_5_results.csv")
    path_patent = os.path.join(pathData, "app_pat_match", "app_pat_match_{}.csv")
    pathMetMet = os.path.join(pathMetrics, "metrics")

    df_cd_final = createCDindex(pathCDI=pathCDI, path_patent=path_patent, yearList=yearList)

    # Use the corrected path in get_file_names()
    for file in get_file_names(pathMetMet):
        print(f"Processing file: {file}")
        
        # try:
        # Assuming extract_year_ipc_vs and other functions are defined
        year_info = extract_year_ipc_vs(file.replace("_aVC", ""))
        print(f"Year info: {year_info}")
        
        df = pd.read_csv(os.path.join(pathMetMet, file))
        # Call functions to generate DataFrames for each metric
        correl = correl_labelScores(df)
        kt = KTcorrel_metrics(df)
        ttest_df = ttest_metric(df)
        rbo_df = rbo_metrics(df, p)
        rL = rL_full(df)
        rL_ind = rL_metricSeparate(df)
        rL_KS = rL_full_aVC(df, "KS")
        rL_ES = rL_full_aVC(df, "ES")
        rL_ind_aVC = rL_metricSeparate_aVC(df)
        cd_cor = cdI_correl(df, df_cd_final)
        
        # Merge DataFrames with blank rows and names
        final_df = merge_dataframes_with_blank_lines([rL, rL_ind, rL_KS, rL_ES, rL_ind_aVC, cd_cor,  correl, ttest_df, kt, rbo_df], 
                                                        ['RL (MLE)', 'RL_ind (MLE)', "rL_KS", "rL_ES", "RL_ind - Controle", "CD-Index & Moyennes", 'Corrélation', "t-test", 'Kendall-Tau',  'RBO'])
        
        # Extract and join the year information to form the sheet name
        sheet_name = ('_'.join(year_info))[:31]  # Ensure the sheet name is within Excel's 31 character limit
        df_list.append(final_df)
        sheet_names.append(sheet_name)  # Use the joined string as the sheet name
    return df_list, sheet_names




##### Concatenate Top 10 ######



def row_bind_selected_files(pathMetrics, target_ipc, target_vs):
    '''
    Concatenate all CSV files from a directory that match a given IPC and VS combination into a single DataFrame.
    Parameters:
        pathMetrics (str): Directory containing the CSV files of outputted scores.
        target_ipc (str): Target IPC code to filter the files (e.g., 'G06F').
        target_vs (str): Target comparison label (e.g., 'top10').
    Returns:
        pd.DataFrame: Concatenated DataFrame of all matching files, with added columns for year, IPC, and VS.
    '''
    file_list=get_file_names(pathMetrics)
    dfs = []

    for file in file_list:
        if not file.endswith("Metrics_aVC.csv"):
            continue  # Skip files that don't end with 'Metrics_aVC.csv'

        year, ipc, vs = extract_year_ipc_vs_top10(file)
        if ipc == target_ipc.upper() and vs == target_vs:
            try:
                df = pd.read_csv(os.path.join(pathMetrics, file))
                df['year'] = year
                df['ipc'] = ipc
                df['vs'] = vs
                dfs.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        print(f"No files found for IPC={target_ipc} and vs={target_vs}")
        return pd.DataFrame()
    

def listMatch(lists, ipc, vs, minMax, col, df_filtered):
    """
    Extracts the top 10 rows for a given metric and appends selected additional columns.

    Parameters:
    -----------
    lists : list of data associated to top 10.
    minMax : "min" or "max"
    ipc : str
        IPC classification or group identifier used as a key in the 'lists' dictionary.
    vs : str
        The view or comparison type, another level of key in the 'lists' dictionary.
    col : str
        The column name used for sorting and selecting the top 10 rows.

    Returns:
    --------
    list of lists
        Each sublist contains:
        [application_number (str), label (int), col value (float), new_ratio, uniq_ratio, diff_ratio, surpDiv_ratio, CD5]
        where the extra columns are pulled from the `df_filtered` DataFrame.
    """
    # The keys you want to keep from df_filtered as the 5 extra columns
    extra_cols = ['new_ratio', 'uniq_ratio', 'diff_ratio', 'surpDiv_ratio', 'CD5']  # replace some_other_col with your actual 5th column name
    top_rows_list = lists[ipc][vs][minMax][col]

    # Filter df_filtered to rows that appear in top_rows_list, for faster lookup
    top_app_nums = [row[0] for row in top_rows_list]
    filtered_df = df_filtered[df_filtered['application_number'].isin(top_app_nums)]

    # Create lookup dict by application_number
    df_lookup = filtered_df.set_index('application_number').to_dict(orient='index')

    # Build final list with 5 extra columns added
    output_list = []
    for app_num, label_val, col_val in top_rows_list:
        row_data = df_lookup[app_num]
        extras = [row_data.get(c, None) for c in extra_cols]
        output_list.append([str(app_num), label_val, col_val] + extras)
    return output_list


def makeLists(pathMetrics, ipc, vs,
              df_CDI=pd.DataFrame,
              pathCDI = "cd5_index_results.csv", 
              path_patent = "patents_{}.csv",
              yearList=range(2012, 2017)):
    '''
    Generate dictionaries of top and bottom 10 values for selected metrics, including cases with missing CD5 values.
    Parameters:
        pathMetrics (str): Path to folder containing metric files.
        ipc (str): IPC code to filter the files (e.g., 'G06F').
        vs (str): Comparison group name (e.g., 'claims_vs_claims_background_Metrics').
        df_CDI (pd.DataFrame or None): Optional precomputed CD-index DataFrame.
        pathCDI (str): Path to the CD-index results file. Default "cd5_index_results.csv"
        path_patent (str): Template path to patent metadata CSVs. Default : "/patents_{}.csv"
        yearList: List of year to make list for
    Returns:
        lists (dict): Nested dictionary of top/bottom 10 values for each metric.
        lists_CD_nan (dict): Same as above but only for rows with missing CD5.
    '''
    # Create nested dictionaries
    lists = {}
    lists_CD_nan = {}
    df = row_bind_selected_files(pathMetrics, ipc, vs)
    df_cd_final = createCDindex(pathCDI, path_patent, df_CDI=df_CDI, yearList=yearList)
    merged = df.merge(df_cd_final, left_on='application_number', right_on='application_number', how="left")
    
    df_filtered = merged[[col for col in merged.columns if col == 'application_number' or col.endswith('ratio') or col == 'CD5' or col=="label"]]
    df_filtered["application_number"] = df_filtered["application_number"].astype(str)

    # Define target columns
    target_cols = [col for col in df_filtered.columns if col.endswith('ratio')] + ['CD5']
    selected_cols = target_cols[:5] + (['CD5'] if 'CD5' not in target_cols[:5] else [])

    df_nan_cd5 = df_filtered[df_filtered['CD5'].isna()]

    # Initialize dicts if not yet done
    lists.setdefault(ipc, {}).setdefault(vs, {"max": {}, "min": {}})
    lists_CD_nan.setdefault(ipc, {}).setdefault(vs, {"max": {}, "min": {}})

    for col in selected_cols:
        # Full data
        lists[ipc][vs]["max"][col] = df_filtered[['application_number', "label", col]].nlargest(10, col).values.tolist()
        lists[ipc][vs]["max"][col] = listMatch(lists, ipc, vs, "max", col, df_filtered)
        lists[ipc][vs]["min"][col] = df_filtered[['application_number', "label", col]].nsmallest(10, col).values.tolist()
        lists[ipc][vs]["min"][col] = listMatch(lists, ipc, vs, "min", col, df_filtered)


        # NaN CD5 subset
        if not df_nan_cd5.empty:
            lists_CD_nan[ipc][vs]["max"][col] = df_nan_cd5[['application_number', "label", col]].nlargest(10, col).values.tolist()
            lists_CD_nan[ipc][vs]["max"][col] = listMatch(lists_CD_nan, ipc, vs, "max", col, df_filtered)
            lists_CD_nan[ipc][vs]["min"][col] = df_nan_cd5[['application_number', "label", col]].nsmallest(10, col).values.tolist()
            lists_CD_nan[ipc][vs]["min"][col] = listMatch(lists_CD_nan, ipc, vs, "min", col, df_filtered)

    return lists, lists_CD_nan


def format_named_df(df, name):
    '''
    Format a DataFrame by inserting title row at the top and appending two blank rows.
    Parameters:
        df (pd.DataFrame): The DataFrame to format.
        name (str): The name to insert in the top-left cell as a section title.
    Returns:
        pd.DataFrame: A new DataFrame with the name row prepended and two empty rows appended.
    '''
    # Rename columns to string in case of numeric column names
    df = df.copy()
    df.columns = df.columns.astype(str)
    
    # Add the name in the top-left cell, and shift the rest down
    header = pd.DataFrame([[name] + [''] * (df.shape[1] - 1)], columns=df.columns)
    df_with_name = pd.concat([header, df], ignore_index=True)
    
    # Add two empty rows (matching column names)
    empty_rows = pd.DataFrame([[''] * df.shape[1]] * 2, columns=df.columns)
    df = pd.concat([df_with_name, empty_rows], ignore_index=True)

    return df

def formatVertical(dfs):
    '''
    Format multiple DataFrames by prepending a title row and appending two blank rows to each, then concatenate them vertically.
    
    Parameters:
        dfs (dict): A dictionary of DataFrames where keys are names and values are DataFrames to be formatted.
    
    Returns:
        pd.DataFrame: A single DataFrame that contains all the formatted DataFrames concatenated vertically.
    '''
    formatted = [format_named_df(df, name) for name, df in dfs.items()]
    final_df = pd.concat(formatted, ignore_index=True)
    return final_df

def bind_horizontal(df1, df2, separator_value=''):
    """
    Horizontally concatenates two dataframes with a separator column in between.
    Fills with NaN if dataframes have unequal lengths.
    
    Parameters:
        df1 (pd.DataFrame): First dataframe
        df2 (pd.DataFrame): Second dataframe
        separator_value (str, optional): Value to insert in separator column. Default is ''.
    
    Returns:
        pd.DataFrame: Concatenated dataframe
    """
    max_len = max(len(df1), len(df2))
    df1 = df1.reindex(range(max_len))
    df2 = df2.reindex(range(max_len))
    separator = pd.DataFrame({ 'sep': [separator_value] * max_len })
    final_df = pd.concat([df1.reset_index(drop=True), separator, df2.reset_index(drop=True)], axis=1)
    return final_df


def top10Complete(pathMetrics,
                  pathData, 
                  ipcList = ["G06F", "A61B", "H01L", "B60L", "E21B", "F03D", "H01L", "H04W", "C07D", "B32B"], 
                  vsList = ["abstract_summary_vs_abstract_summary_background_Metrics", "claims_vs_claims_background_Metrics"],
                  yearList = range(2012,2017)):
    '''
    Generate and write the top 10 values for various patent metrics (max and min) for different IPC and vs codes, 
    then return a vertically concatenated DataFrame containing all results.
    
    Parameters:
        pathMetrics (str): Path to the metrics directory (containing metrics)
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab).
        ipcList (list): List of IPC codes to process.
        vsList (list): List of vs metrics to process.
        yearList (list): List of years to process.
        
    Returns:
        pd.DataFrame: A DataFrame containing the concatenated results for all IPC and vs combinations.
    '''

    pathOutputMinMax = os.path.join(pathMetrics, "top10")
    pathMetMet = os.path.join(pathMetrics, "metrics")
    pathCDI = os.path.join(pathMetrics, "cd_index", "cd_index_5_results.csv")
    path_patent = os.path.join(pathData, "app_pat_match", "app_pat_match_{}.csv")

    dfs = defaultdict(lambda: defaultdict(pd.DataFrame))

    df_CDI = pd.read_csv(pathCDI)


    for vs in vsList:
        print(vs)
        for ipc in ipcList:
            print(ipc)
            lists, lists_CD_nan = makeLists(pathMetMet, ipc, vs, df_CDI, pathCDI, path_patent, yearList= yearList)

            # Create well ordered dataframes
            new_ratio_max = lists[ipc][vs]["max"]["new_ratio"]
            uniq_ratio_max = lists[ipc][vs]["max"]["uniq_ratio"]
            diff_ratio_max = lists[ipc][vs]["max"]["diff_ratio"]
            surp_ratio_max = lists[ipc][vs]["max"]["surpDiv_ratio"]
            cd5_max = lists[ipc][vs]["max"]["CD5"]

            new_ratio_min = lists[ipc][vs]["min"]["new_ratio"]
            uniq_ratio_min = lists[ipc][vs]["min"]["uniq_ratio"]
            diff_ratio_min = lists[ipc][vs]["min"]["diff_ratio"]
            surp_ratio_min = lists[ipc][vs]["min"]["surpDiv_ratio"]
            cd5_min = lists[ipc][vs]["min"]["CD5"]

            new_ratio_max_nan = lists_CD_nan[ipc][vs]["max"]["new_ratio"]
            uniq_ratio_max_nan = lists_CD_nan[ipc][vs]["max"]["uniq_ratio"]
            diff_ratio_max_nan = lists_CD_nan[ipc][vs]["max"]["diff_ratio"]
            surp_ratio_max_nan = lists_CD_nan[ipc][vs]["max"]["surpDiv_ratio"]
            cd5_max_nan = lists_CD_nan[ipc][vs]["max"]["CD5"]

            new_ratio_min_nan = lists_CD_nan[ipc][vs]["min"]["new_ratio"]
            uniq_ratio_min_nan = lists_CD_nan[ipc][vs]["min"]["uniq_ratio"]
            diff_ratio_min_nan = lists_CD_nan[ipc][vs]["min"]["diff_ratio"]
            surp_ratio_min_nan = lists_CD_nan[ipc][vs]["min"]["surpDiv_ratio"]
            cd5_min_nan = lists_CD_nan[ipc][vs]["min"]["CD5"]


            dfs_max = {"new_ratio_max":pd.DataFrame(new_ratio_max, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "uniq_ratio_max":pd.DataFrame(uniq_ratio_max, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "diff_ratio_max":pd.DataFrame(diff_ratio_max, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "surp_ratio_max":pd.DataFrame(surp_ratio_max, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "cd5_max":pd.DataFrame(cd5_max, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"])}
            dfs_max_nan = {"new_ratio_max_nan":pd.DataFrame(new_ratio_max_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "uniq_ratio_max_nan":pd.DataFrame(uniq_ratio_max_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "diff_ratio_max_nan":pd.DataFrame(diff_ratio_max_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "surp_ratio_max_nan":pd.DataFrame(surp_ratio_max_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "cd5_max_nan":pd.DataFrame(cd5_max_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"])}

            dfs_min = {"new_ratio_min":pd.DataFrame(new_ratio_min, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "uniq_ratio_min":pd.DataFrame(uniq_ratio_min, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "diff_ratio_min":pd.DataFrame(diff_ratio_min, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "surp_ratio_min":pd.DataFrame(surp_ratio_min, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "cd5_min":pd.DataFrame(cd5_min, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"])}
            dfs_min_nan = {"new_ratio_min_nan":pd.DataFrame(new_ratio_min_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "uniq_ratio_min_nan":pd.DataFrame(uniq_ratio_min_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "diff_ratio_min_nan":pd.DataFrame(diff_ratio_min_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "surp_ratio_min_nan":pd.DataFrame(surp_ratio_min_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"]),
                        "cd5_min_nan":pd.DataFrame(cd5_min_nan, columns=["application_number", "label", "score", "new_ratio", "uniq_ratio", "diff_ratio", "surp_ratio", "CD5"])}
            

            final_df_max = formatVertical(dfs_max)
            final_df_max_nan = formatVertical(dfs_max_nan)
            final_df_min = formatVertical(dfs_min)
            final_df_min_nan = formatVertical(dfs_min_nan)

            complete_max = bind_horizontal(final_df_max, final_df_max_nan)
            complete_min = bind_horizontal(final_df_min, final_df_min_nan)

            complete_ipc = bind_horizontal(complete_max, complete_min)

            # Add ipc and vs to the complete_ipc
            new_rows = pd.DataFrame(
                [[ipc] + [""] * complete_ipc.shape[1], [vs] + [""] * complete_ipc.shape[1]],
                columns=[""] + list(complete_ipc.columns)
            )

            # Step 2: Add an empty column to the original complete_ipc
            complete_ipc_with_col = complete_ipc.copy()
            complete_ipc_with_col.insert(0, "", [""] * len(complete_ipc))

            # Step 3: Concatenate the new rows on top
            result = pd.concat([new_rows, complete_ipc_with_col], ignore_index=True)

            dfs[ipc][vs] = result  # use a meaningful key

        
    dfs_ipc = {}

    for ipc in ipcList:

        # Start with the first vs to initialize
        df_combined = dfs[ipc][vsList[0]]
        for vs in vsList[1:]:
            df_combined = bind_horizontal(df_combined, dfs[ipc][vs], separator_value='')
        dfs_ipc[ipc] = df_combined

    final_df = formatVertical(dfs_ipc)
    final_df.columns = ["" if col == "sep" else col for col in final_df.columns]
    final_df.to_csv(os.path.join(pathOutputMinMax, "top10.csv"), index=False)

            
            


##### Iterate over metrics ######


def compute_scores(KB_matrix=None, KB_dist=None, NewKB_dist=None, variation_dist=None, 
                   dict_know_pmi=None, EB_PMI=None, base_bigram_set=None, New_EB_PMI=None,
                   newness_type='div', uniq_type='dist', diff_type='local', neighbor_dist=0.,
                   thr_new_div=0.0041, thr_new_div_flag=0.0014, thr_new_prob=57.14, thr_new_prob_flag=0.0014, thr_uniq_flag =0.527,
                    thr_uniqp_flag=0.1295, thr_diff =0.4177, thr_surp=0.00256,
                   useClusters=False, KSCluster=None, nb_clusters=4, metrics_to_compute=None):
    """
    Compute multiple innovation-related metrics (newness, uniqueness, difference, and surprise) for evaluating 
    linguistic or conceptual novelty in a knowledge base over time.

    Parameters:
        KB_matrix (array-like): Co-occurrence or term matrix of the known knowledge base.
        KB_dist (array-like): Distribution of known terms or concepts.
        NewKB_dist (array-like): Distribution of new or recent terms/concepts.
        variation_dist (array-like): Vector representing the difference between old and new distributions.
        dict_know_pmi (dict): PMI values for the known terms.
        EB_PMI (dict): PMI values from the established knowledge base.
        base_bigram_set (set): Set of base bigrams to compare against.
        New_EB_PMI (dict): PMI values for the new period to evaluate surprise.
        newness_type (str): Method for newness detection ('div' for divergence, 'prob' for probability-based).
        uniq_type (str): Method for uniqueness ('dist' for distance-based, 'proto' for shift in prototype).
        diff_type (str): Method for difference computation ('local' or 'global').
        neighbor_dist (float): Precomputed neighbor distance, set to 0 to estimate internally.
        thr_new_div (float): Threshold for JS divergence to detect significant distribution change.
        thr_new_div_flag (float): Threshold for divergence-based novelty flagging.
        thr_new_prob (float): Threshold for probability mass when using newness probability method.
        thr_new_prob_flag (float): Threshold to flag novelty based on probability.
        thr_uniq_flag (float): Threshold for uniqueness flag using distance-based method.
        thr_uniqp_flag (float): Threshold for uniqueness flag using prototype shift method.
        thr_diff (float): Threshold for difference score binarization.
        thr_surp (float): Threshold to flag surprising bigrams based on PMI differences.
        useClusters (bool): Whether to use clustering (KMeans) in difference score computation.
        KSCluster (object): A KSCluster instance for cluster-based difference methods.
        nb_clusters (int): Number of clusters to use if clustering is enabled.
        metrics_to_compute (list or None): List of metrics to compute from ["newness", "uniqueness", "difference", "surprise"].

    Returns:
        tuple: (
            newness (float or None),           # Newness score
            novelty_new (int or None),         # Binary flag if newness > threshold
            uniqueness (float or None),        # Uniqueness score
            novelty_uniq (int or None),        # Binary flag if uniqueness > threshold
            dif_score (float or None),         # Difference score
            dif_bin (int or None),             # Binary flag if difference > threshold
            neighbor_dist (float),             # Distance used for neighbor-based comparisons
            mean100 (float or None),           # Mean of top 100 ratios (local difference)
            dist_surprise (int or None),       # Count of surprising distributional bigrams
            uniq_surprise (int or None)        # Count of unique surprising bigrams
        )

    Notes:
        - This function supports multiple methods for detecting conceptual innovation and can combine them.
        - Useful for evaluating shifts in language, topic modeling, or knowledge base evolution over time.
        - All returned values may be None if their respective metric is not included in `metrics_to_compute`.
    """
    
    if metrics_to_compute is None:
        metrics_to_compute = ["newness", "uniqueness", "difference", "surprise"]
    
    newness = novelty_new = None
    uniqueness = novelty_uniq = None
    dif_score = dif_bin = mean100 = None
    dist_surprise = uniq_surprise = None

    if "newness" in metrics_to_compute:
        newness_calc = Newness(KB_dist, variation_dist)
        if newness_type == 'div':
            newness, novelty_new = newness_calc.divergent_terms(thr_div=thr_new_div, thr_new=thr_new_div_flag)
        else:
            newness, novelty_new = newness_calc.probable_terms(thr_prob=thr_new_prob, thr_new= thr_new_prob_flag)

    if "uniqueness" in metrics_to_compute:
        uniqueness_calc = Uniqueness(KB_dist)
        if uniq_type == 'dist':
            uniqueness, novelty_uniq = uniqueness_calc.dist_to_proto(variation_dist, thr_uniq=thr_uniq_flag)
        else:
            uniqueness, novelty_uniq = uniqueness_calc.proto_dist_shift(NewKB_dist, thr_uniqp=thr_uniqp_flag)

    if "difference" in metrics_to_compute:
        if useClusters:
            if neighbor_dist == 0. and KSCluster is not None:
                neighbor_dist = KSCluster.dist_estimate_clusters(iterations=256, nb_clusters=nb_clusters)
            if diff_type == "local":
                dif_score, dif_bin, mean100 = KSCluster.ratio_to_neighbors_kmeans(
                    variation_dist=variation_dist, nb_clusters=nb_clusters, neighbor_dist=neighbor_dist)
        else:
            difference = Difference(KB_matrix, variation_dist, N=3)
            if neighbor_dist == 0.:
                neighbor_dist = difference.dist_estimate()
            if diff_type == 'global':
                dif_score, dif_bin = difference.ratio_to_all(neighbor_dist, thr_diff=thr_diff)
                mean100 = None
            else:
                dif_score, dif_bin, mean100 = difference.ratio_to_neighbors_joblib(neighbor_dist, thr_diff=thr_diff)

    if "surprise" in metrics_to_compute:
        surprise_calc = Surprise(New_EB_PMI)
        dist_surprise, uniq_surprise = surprise_calc.unique_surp_courte(New_EB_PMI, EB_PMI, base_bigram_set, eps=0.00, thr_surp=thr_surp)

    return newness, novelty_new, uniqueness, novelty_uniq, dif_score, dif_bin, neighbor_dist, mean100, dist_surprise, uniq_surprise


def iterateMetrics(path, pathOutput, tE_cols, base_cols, w_size, year, ipc, chunksize = 10000, metrics_to_compute = ["newness", "uniqueness", "difference", "surprise"],
                   thr_new_div=0.0041, thr_new_div_flag=0.0014, thr_new_prob=57.14, thr_new_prob_flag=0.0014, thr_uniq_flag =0.527, thr_uniqp_flag=0.1295,
                    useClusters = True, nb_clusters=4, neighbor_dist=0., thr_diff =0.85, thr_surp=0.00256, forDemo=False):
    """
    Compute novelty and surprise metrics between a set of test embeddings (tE) and base knowledge (KS/ES) using PMI and distributional shifts.

    This function:
    - Loads cleaned datasets (tE, KS, ES) for a given year and IPC.
    - Transforms these datasets by combining specified columns.
    - Computes PMI scores for the base corpus (ES) in chunks to manage memory.
    - Constructs a base bigram PMI distribution and initializes reference distributions.
    - Iteratively, for each test entry in tE:
        * Updates the PMI using the new entry.
        * Computes variation in the KS distribution with and without the new entry.
        * Computes novelty, uniqueness, divergence, and surprise metrics.
    - Optionally clusters the KS distribution for neighborhood comparisons.
    - Stores all computed metrics into a DataFrame and writes it to a CSV.

    Parameters:
    ----------
    path : str
        Path to the directory containing input files (`tE`, `KS`, `ES` subfolders).
    
    pathOutput : str
        Path to save the resulting metrics CSV.
    
    tE_cols : list of str
        List of column names in tE to combine into the input text.
    
    base_cols : list of str
        List of column names in KS and ES to combine into the base corpus.
    
    w_size : int
        Window size for PMI calculations (not directly used here, possibly for `OptimizedIncrementalPMI`).
    
    useClusters : bool
        If True, use clustering on KS to compute cluster-aware distances.
    
    year : int or str
        Year of the data to load.
    
    ipc : str
        IPC code used to name the input files.
    
    chunksize : int, optional
        Size of text chunks to use when computing PMI on ES data (default is 10,000).

    metrics_to_compute : list of str, optional
        List of metric categories to compute. Could be any of ["newness", "uniqueness", "difference", "surprise"].
    thr_new_div : float
         Divergence threshold. Default 0.0041, could be changed for each IPC class & year (for divergent_terms) 
    thr_new_div_flag : float 
        Newness threshold to flag novelty for divergent_terms (for divergent_terms). Default 0.0014
    thr_new_prob : float
        Probability ratio threshold to identify significant terms. Default 57.14, could be changed for each IPC class & year (for probable_terms) 
    thr_new_prob_flag : float
        Newness threshold to flag novelty for probable_terms (newness) (for probable_terms). Default 0.0014
    thr_uniq_flag : float
        Threshold to consider the distribution unique (for dist_to_proto). Default 0.527
    thr_uniqp_flag : float
        Threshold to flag distribution shift (for proto_dist_shift). Default 0.1295
    useClusters : bool
        Use clusters or not for computing neighborhood distance and difference metric. Default True
    nb_clusters : int
        number of clusters yo use. default 4
    neighbor_dist : float
        Set neighborhood distance. Default to 0. will compute the neighborhood distance for year and ipc code
    thr_diff : float
        Threshold to flag difference, default 0.85
    thr_surp : float
        Threshold above which the distribution is considered surprising. Defualt 0.00256
    forDemo : boolean
        Trueruns only for one patent (for the demo)
    
    Returns:
    -------
    None
        The function saves a CSV file with various computed metrics such as:
        - new_ratio, new_bin
        - uniq_ratio, uniq_bin
        - diff_ratio, diff_bin
        - neighboroud_distance
        - surpDiv_ratio, surpDiv_bin
    """
    start_time = time.time()  # Record the starting time
    print(f"{year}")
    print(f"    {ipc}")
    tE = pd.read_csv(os.path.join(path, "tE", f"{year}_{ipc}_tE_cleaned.csv"))
    KS = pd.read_csv(os.path.join(path, "KS", f"{year}_{ipc}_KS_cleaned.csv"))
    ES = pd.read_csv(os.path.join(path, "ES", f"{year}_{ipc}_ES_cleaned.csv"))

    ### Transforming KB into distribution
    application_number = tE["application_number"]
    label = tE["label"]

    tE=combine_columns(tE, tE_cols) 
    KS=combine_columns(KS, base_cols) 
    ES=combine_columns(ES, base_cols)


    if "surprise" in metrics_to_compute:
        print("ES")
        chunk_size = chunksize  # Adjust based on available memory
        ES_PMI = OptimizedIncrementalPMI(3)

        for i in range(0, len(ES), chunk_size):
            base_texts = [word for text in (ES[i:i+chunk_size]) for word in text.split()]
            ES_PMI.update(base_texts)

        instance_ES_pmi = ES_PMI.compute_pmi()
        base_bigram_set = set(instance_ES_pmi.keys())

        dict_ES_pmi = pmi_to_dict_adj_dict(instance_ES_pmi)

        print("ES finidhsed")

    if any(m in metrics_to_compute for m in ["newness", "uniqueness", "difference"]):
        print("KS")
        KS_matrix, KS_dist, KS_Count_matrix = docs_distribution(baseSpace=KS, tE=tE)
        KS_size = list(range(KS_matrix.shape[0]))
        print("KS finished")

    if "difference" in metrics_to_compute:
        if useClusters==True:
            KSClusterDiff1000 = ClusterKS(list_know_P=KS_matrix, new_Q= None, N=100, nbPtsPerCluster=1000)
            KSClusterDiff1000.clusterKS()

    neighborhood_distance  = neighbor_dist
    new_ratio_vec = []
    new_bin_vec = []
    uniq_ratio_vec = []
    uniq_bin_vec = []
    diff_ratio_vec = []
    diff_bin_vec = []
    neighborhood_distance_vec = []
    surpDiv_ratio_vec = []
    surpDiv_bin_vec = []
    
    if forDemo==True:
        range_ = range(1,2)
        application_number = application_number[list(range_)[0]]
        label = label[list(range_)[0]]
    else:
        range_ = range(len(tE))
    for i in range_:
        if any(m in metrics_to_compute for m in ["newness", "uniqueness", "difference"]):
            select_variation = KS_size + [len(KS_size)+i]
            NewKS_dist, variation_dist = new_distribution(KS_Count_matrix, select_variation)
    
        if "surprise" in metrics_to_compute:
            baseTexts_update = [tE[i]]
            update_text = [word for text in (baseTexts_update) for word in text.split()]
            new_pmi = OptimizedIncrementalPMI(window_size=3)
            new_pmi.update(update_text)
            newpmi_PMI = new_pmi.compute_pmi()
            
        # print("compute scores")
        if useClusters==True:
            score_args = {
                "metrics_to_compute": metrics_to_compute,
                "thr_new_div": thr_new_div,
                "thr_new_div_flag": thr_new_div_flag,
                "thr_new_prob": thr_new_prob,
                "thr_new_prob_flag": thr_new_prob_flag,
                "thr_uniq_flag": thr_uniq_flag,
                "thr_uniqp_flag": thr_uniqp_flag,
                "thr_diff": thr_diff,
                "thr_surp": thr_surp,
                "useClusters": useClusters,
                "nb_clusters": nb_clusters
            }
            if "surprise" in metrics_to_compute:
                score_args.update({
                    "EB_PMI": instance_ES_pmi,
                    "base_bigram_set": base_bigram_set,
                    "dict_know_pmi": dict_ES_pmi,
                    "New_EB_PMI": newpmi_PMI
                })

            if any(metric in metrics_to_compute for metric in ["newness", "uniqueness", "difference"]):
                score_args.update({
                    "KB_matrix": KS_matrix,
                    "KB_dist": KS_dist,
                    "NewKB_dist": NewKS_dist,
                    "variation_dist": variation_dist,
                    "neighbor_dist": neighbor_dist,
                    "useClusters": useClusters
                })

            if "difference" in metrics_to_compute and useClusters:
                score_args["KSCluster"] = KSClusterDiff1000

            results = compute_scores(**score_args)
            # results = compute_scores(KB_matrix=KS_matrix, KB_dist=KS_dist, NewKB_dist=NewKS_dist, variation_dist=variation_dist, 
            #                         EB_PMI=instance_ES_pmi, base_bigram_set=base_bigram_set, dict_know_pmi=dict_ES_pmi, New_EB_PMI=newpmi_PMI,
            #                         neighbor_dist=neighborhood_distance, useClusters=True, KSCluster=KSClusterDiff1000)
        else:
            score_args = {
                "metrics_to_compute": metrics_to_compute,
                "thr_new_div": thr_new_div,
                "thr_new_div_flag": thr_new_div_flag,
                "thr_new_prob": thr_new_prob,
                "thr_new_prob_flag": thr_new_prob_flag,
                "thr_uniq_flag": thr_uniq_flag,
                "thr_uniqp_flag": thr_uniqp_flag,
                "thr_diff": thr_diff,
                "thr_surp": thr_surp,
                "useClusters": useClusters,
                "nb_clusters": nb_clusters
            }

            if "surprise" in metrics_to_compute:
                score_args.update({
                    "EB_PMI": instance_ES_pmi,
                    "base_bigram_set": base_bigram_set,
                    "dict_know_pmi": dict_ES_pmi,
                    "New_EB_PMI": newpmi_PMI
                })

            if any(metric in metrics_to_compute for metric in ["newness", "uniqueness", "difference"]):
                score_args.update({
                    "KB_matrix": KS_matrix,
                    "KB_dist": KS_dist,
                    "NewKB_dist": NewKS_dist,
                    "variation_dist": variation_dist,
                    "neighbor_dist": neighborhood_distance,
                    "useClusters": useClusters
                })

            # if "difference" in metrics_to_compute and useClusters:
            #     score_args["KSCluster"] = KSClusterDiff1000

            results = compute_scores(**score_args)
            # results = compute_scores(KB_matrix=KS_matrix, KB_dist=KS_dist, NewKB_dist=NewKS_dist, variation_dist=variation_dist, 
            #                         EB_PMI=instance_ES_pmi, base_bigram_set=base_bigram_set, dict_know_pmi=dict_ES_pmi, New_EB_PMI=newpmi_PMI,
            #                         neighbor_dist=neighborhood_distance, useClusters=useClusters)


        if "newness" in metrics_to_compute:
            new_ratio = results[0]
            new_bin = results[1]
            new_ratio_vec.append(new_ratio)
            new_bin_vec.append(new_bin)

        if "uniqueness" in metrics_to_compute:
            uniq_ratio = results[2]
            uniq_bin = results[3]
            uniq_ratio_vec.append(uniq_ratio)
            uniq_bin_vec.append(uniq_bin)

        if "difference" in metrics_to_compute:
            diff_ratio = results[4]
            diff_bin = results[5]
            neighborhood_distance = results[6] #update neighborhood distance so it isn't reset
            diff_ratio_vec.append(diff_ratio)
            diff_bin_vec.append(diff_bin)
            neighborhood_distance_vec.append(neighborhood_distance)

        if "surprise" in metrics_to_compute:
            surpDiv_ratio = results[8]
            surpDiv_bin = results[9]
            surpDiv_ratio_vec.append(surpDiv_ratio)
            surpDiv_bin_vec.append(surpDiv_bin)

        # Time for each thousand iteration
        if i % 1000 == 0:
            current_time = time.time()  # Get the current time
            elapsed_time = current_time - start_time  # Calculate elapsed time
            if i % 1000 == 0:
                print(f"{i} on {len(tE)}, time since last print: {elapsed_time:.2f} seconds")
            start_time = current_time  # Reset the start time for the next interval
                # Reset the start time to measure time per iteration
            start_time = time.time()

    
    start_time = time.time()
    df_dict = {
        "application_number": application_number,
        "label": label
    }

    if "newness" in metrics_to_compute:
        df_dict.update({
            "new_ratio": new_ratio_vec,
            "new_bin": new_bin_vec
        })

    if "uniqueness" in metrics_to_compute:
        df_dict.update({
            "uniq_ratio": uniq_ratio_vec,
            "uniq_bin": uniq_bin_vec
        })

    if "difference" in metrics_to_compute:
        df_dict.update({
            "neighborhood_distance": neighborhood_distance_vec,
            "diff_ratio": diff_ratio_vec,
            "diff_bin": diff_bin_vec
        })

    if "surprise" in metrics_to_compute:
        df_dict.update({
            "surpDiv_ratio": surpDiv_ratio_vec,
            "surpDiv_bin": surpDiv_bin_vec
        })

    df = pd.DataFrame(df_dict)

    if len(metrics_to_compute) == 4:
        metrics = "Metrics"
    else:
        metrics = "".join([metric[0].upper() for metric in metrics_to_compute])
    tE_cols_str = "_".join(tE_cols)
    base_cols_str = "_".join(base_cols)
    df.to_csv(pathOutput + f'/{year}_{ipc}_{tE_cols_str}_vs_{base_cols_str}_{metrics}.csv', index=False)   


def measureNov(pathData, pathMetrics, tE_cols, base_cols, w_size, yearList, ipcList, chunksize=10000,  metrics_to_compute = ["newness", "uniqueness", "difference", "surprise"],
               thr_new_div=0.0041, thr_new_div_flag=0.0014, thr_new_prob=57.14, thr_new_prob_flag=0.0014, thr_uniq_flag =0.527, thr_uniqp_flag=0.1295,
                useClusters = True, nb_clusters=4, neighbor_dist=0., thr_diff =0.85, thr_surp=0.00256, forDemo=False):
    """
    Computes novelty-related metrics across multiple (year, IPC) combinations.

    This function loads data files from the specified path, extracts available
    (year, IPC) pairs if not explicitly provided, and iteratively calls `iterateMetrics`
    to compute novelty and related metrics (e.g., newness, uniqueness, difference, surprise)
    for each pair.

    Parameters:
    pathData : str
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab).
    pathMetrics : str
        Path to the metrics directory (containing metrics)
    tE_cols : list of str
        Column names to extract from the tE dataset.
    base_cols : list of str
        Column names used to define the base bigram or concept set.
    w_size : int
        Window size for co-occurrence computations.
    useClusters : bool
        Whether to compute difference metrics using clustering (True) or not (False).
    yearList : list of str
        List of years to process. If empty, they are inferred from file names.
    ipcList : list of str
        List of IPC codes to process. If empty, they are inferred from file names.
    chunksize : int, optional
        Size of text chunks to use when computing PMI on ES data (default is 10,000).
    metrics_to_compute : list of str, optional
        List of metric categories to compute. Could be any of ["newness", "uniqueness", "difference", "surprise"].
    thr_new_div : float
         Divergence threshold. Default 0.0041, could be changed for each IPC class & year (for divergent_terms) 
    thr_new_div_flag : float 
        Newness threshold to flag novelty for divergent_terms (for divergent_terms). Default 0.0014
    thr_new_prob : float
        Probability ratio threshold to identify significant terms. Default 57.14, could be changed for each IPC class & year (for probable_terms) 
    thr_new_prob_flag : float
        Newness threshold to flag novelty for probable_terms (newness) (for probable_terms). Default 0.0014
    thr_uniq_flag : float
        Threshold to consider the distribution unique (for dist_to_proto). Default 0.527
    thr_uniqp_flag : float
        Threshold to flag distribution shift (for proto_dist_shift). Default 0.1295
    useClusters : bool
        Use clusters or not for computing neighborhood distance and difference metric. Default True
    nb_clusters : int
        number of clusters yo use. default 4
    neighbor_dist : float
        Set neighborhood distance. Default to 0. will compute the neighborhood distance for year and ipc code
    thr_diff : float
        Threshold to flag difference, default 0.85
    thr_surp : float
        Threshold above which the distribution is considered surprising. Defualt 0.00256
    forDemo : boolean
        Trueruns only for one patent (for the demo)

    Notes:
    - Assumes that input file names contain year and IPC information in a consistent format.
    - Assumes the subdirectories `tE`, `KS`, and `ES` contain aligned sets of files.
    - Validates that all three datasets (tE, KS, ES) contain matching (year, IPC) pairs.
    """

    pathClean = os.path.join(pathData, "csv_clean")
    pathMetMet = os.path.join(pathMetrics, "metrics_raw")

    if yearList == [] or ipcList == []:
        tE_names = get_file_names(pathClean+"tE")
        KS_names = get_file_names(pathClean+"KS")
        ES_names = get_file_names(pathClean+"ES")

        tE_set = set([extract_year_ipc(string) for string in tE_names])
        KS_set = set([extract_year_ipc(string) for string in KS_names])
        ES_set = set([extract_year_ipc(string) for string in ES_names])

        assert(tE_set==KS_set==ES_set)

        yearList = sorted(list(set([year for year, ipc in list(tE_set)])))
        ipcList = list(set([ipc for year, ipc in list(tE_set)]))
        print(yearList)
        print(ipcList)

    for year in yearList:
        for ipc in ipcList:
            iterateMetrics(pathClean, pathMetMet, tE_cols, base_cols, w_size, year, ipc, chunksize=chunksize, metrics_to_compute=metrics_to_compute,
                            thr_new_div=0.0041, thr_new_div_flag=0.0014, thr_new_prob=57.14, thr_new_prob_flag=0.0014, thr_uniq_flag =0.527, thr_uniqp_flag=0.1295,
                            useClusters = True, nb_clusters=4, neighbor_dist=0., thr_diff =0.85, thr_surp=0.00256,forDemo=forDemo)
            


