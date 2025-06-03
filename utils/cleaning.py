import spacy
nlp = spacy.load("en_core_web_sm")
import pandas as pd
from tqdm import tqdm
import nltk
import re
from nltk.corpus import stopwords
from collections import Counter
nltk.download('stopwords')
import glob
import os
import random
import ast
tqdm.pandas(miniters=100000, maxinterval=1000)

class Vocab():
    """
    Class to create Vocab
    Parameters:
        technet
        stopwords
        df_lemm : lemmatized technet
    """
    
    def __init__(self, technet, df_lemm, stopwords):
        self.technet = technet
        self.set_vocab = self.setVocab()
        self.stopwords = stopwords
        self.tn_lemm_filtered = self.filterSW(df_lemm)
        self.lemm_dict = pd.Series(self.tn_lemm_filtered['lemmatized'].values, index=self.tn_lemm_filtered['technet_vocab']).to_dict()
        

    def setVocab(self):
        '''
        Extract unique vocabulary terms from technet.
        Parameters: None (uses self.technet)
        Returns:
            set_keep : Set of vocabulary terms
        '''
        keep_vocab = list(self.technet['technet_vocab'])
        set_keep = set(keep_vocab)
        return set_keep
    
    def clean_tokens(self, text):
        '''
        Clean input text by keeping only tokens present in the vocabulary.
        Parameters:
            text : str
                Raw text to be tokenized and filtered.
        Returns:
            filtered_text : str
                Space-separated string of tokens found in the vocabulary.
        '''
        doc_token = []
        for tok in nlp.tokenizer(str(text)):
            token_lower = tok.text.lower()
            if token_lower in self.set_vocab:
                doc_token.append(token_lower)
        filtered_text = ' '.join(doc_token)
        return filtered_text
    
    def cleanDF(self, df_text, type="all"):
        '''
        Apply vocabulary-based token cleaning to specified sections of a patent DataFrame.
        Parameters:
            df_text : DataFrame
                DataFrame containing patent text with fields 'background', 'claims', 'abstract', and/or 'summary'.
            type : str, default="all"
                Indicates which fields to clean:
                - "all": clean all sections
                - "claims": clean only the claims section
                - "others": clean background, abstract, and summary
        Returns:
            df_clean : DataFrame
                DataFrame with cleaned text fields and original metadata (application_number, label).
        '''
        # applies cleaning to all texts in patents.
        df_clean = pd.DataFrame()
        df_clean["application_number"] = df_text["application_number"]
        df_clean["label"] = df_text["label"]
        print("Clean")
        if type=="all":
            df_clean['background'] = df_text['background'].progress_apply(self.clean_tokens)
            df_clean['claims'] = df_text['claims'].progress_apply(self.clean_tokens)
            df_clean['abstract'] = df_text['abstract'].progress_apply(self.clean_tokens)
            df_clean['summary'] = df_text['summary'].progress_apply(self.clean_tokens)
        elif type=="claims":
            df_clean['claims'] = df_text['claims'].progress_apply(self.clean_tokens)
        elif type=="others":
            df_clean['background'] = df_text['background'].progress_apply(self.clean_tokens)
            df_clean['abstract'] = df_text['abstract'].progress_apply(self.clean_tokens)
            df_clean['summary'] = df_text['summary'].progress_apply(self.clean_tokens)
        return df_clean
    
    def lemmatize_with_dict(self, text):
        """
        Lemmatize input text using a predefined dictionary.

        Each word in the input text is looked up in the `lemm_dict` attribute.
        Only words that have a corresponding lemma in the dictionary are kept.

        Parameters:
            text (str): A space-separated string of words to be lemmatized.

        Returns:
            str: A space-separated string of lemmatized words, including only words found in the dictionary.
        """
        # lemmatize technet
        words = text.split()
        lemmatized_text = []

        for word in words:
            # Check if the word is in the dictionary and add it only if found
            lemmatized_word = self.lemm_dict.get(word)

            if isinstance(lemmatized_word, str):
                lemmatized_text.append(lemmatized_word)

        return ' '.join(lemmatized_text)
    
    def lemmDF(self, df_clean):
        """
        Lemmatize all text columns in a cleaned DataFrame.

        Applies lemmatization to each cell in all columns except 
        'application_number' and 'label', using a predefined 
        lemmatization dictionary.

        Parameters:
            df_clean (pd.DataFrame): DataFrame with cleaned text data.

        Returns:
            pd.DataFrame: DataFrame with lemmatized text in applicable columns.
        """
        # Lemmatize the technet cleaned df
        print("Lemmatize")
        lemm_df = df_clean
        for column in df_clean.columns:
            if column not in ["application_number", "label"]:
                lemm_df[column] = df_clean[column].progress_apply(lambda x: self.lemmatize_with_dict(str(x)))
        return lemm_df
    
    def filterSW(self, df_lemm):
        """
        Filter out rows containing stopwords from a lemmatized DataFrame.

        Removes rows where the 'lemmatized' column contains terms found in 
        the stopword list stored in self.stopwords.

        Parameters:
            df_lemm (pd.DataFrame): DataFrame with a 'lemmatized' column.

        Returns:
            pd.DataFrame: Filtered DataFrame with stopwords removed.
        """
        tn_lemm = df_lemm[~df_lemm['lemmatized'].isin(self.stopwords)]
        return(tn_lemm)


# def get_file_names(pathCSV):
#     # Get the list of files in the directory
#     all_files = [f for f in os.listdir(pathCSV) if os.path.isfile(os.path.join(pathCSV, f))]
#     # Filter the files for those ending with '.csv'
#     csv_files = [f for f in all_files if f.endswith('.csv')]
#     return csv_files

def get_file_names(pathCSV):
    """
    Get a list of CSV file names in a given directory.

    Parameters:
        pathCSV (str): Path to the directory to search for CSV files.

    Returns:
        list of str: List of file names ending with '.csv' in the specified directory.
                     If the directory does not exist or contains no CSV files, returns an empty list.
    """
    if not os.path.exists(pathCSV):
        print(f"Warning: Directory '{pathCSV}' not found.")
        return []

    all_files = [f for f in os.listdir(pathCSV) if os.path.isfile(os.path.join(pathCSV, f))]
    csv_files = [f for f in all_files if f.endswith('.csv')]

    if not csv_files:
        print(f"Warning: No CSV files found in '{pathCSV}'.")

    return csv_files


def extract_year_ipc(filename):
    """
    Extract the year and IPC code from a CSV filename.

    Supports multiple filename formats such as:
        - '2020_H01L_patents_toEval.csv'
        - '2020_2022_H01L_KS_raw.csv'
        - '2020_H01L.csv'

    Uses a regular expression to capture:
        - A 4-digit year at the beginning (or first of two years),
        - A 4-letter IPC code,
        - Ignores any additional components after the IPC code.

    Parameters:
        filename (str): The filename to parse.

    Returns:
        tuple: (year, ipc) if the format matches, otherwise (None, None).
               Prints a warning if the filename doesn't match the expected pattern.
    """
    # Regular expression to match both formats
    match = re.match(r'(\d{4})(?:_\d{4})?_(\w{4})(?:_.*)?\.csv', filename)
    
    if match:
        year = match.group(1)
        ipc = match.group(2)
        return year, ipc
    else:
        print(f"Filename format unexpected: {filename}")
        return None, None

def extract_year_ipc_vs(filename):
    """
    Extract year, IPC code, and variant string from a filename.

    The function expects filenames in the format:
    'YYYY_IPCC_variantString_Metrics.csv', where:
        - YYYY is a 4-digit year,
        - IPCC is a 4-letter IPC code,
        - variantString is an arbitrary string (can contain underscores),
        - and the filename ends with '_Metrics.csv'.

    Parameters:
        filename (str): The filename to parse.

    Returns:
        tuple: (year, ipc, variantString) if the format matches, otherwise (None, None, None).
               Prints a warning if the format is unexpected.
    """
    # Regular expression to capture year, ipc, and the part between ipc and Metrics
    match = re.match(r'(\d{4})_(\w{4})_([^_]+_.*)_Metrics\.csv', filename)
    
    if match:
        year = match.group(1)
        ipc = match.group(2)
        vs = match.group(3)
        return year, ipc, vs
    else:
        print(f"Filename format unexpected: {filename}")
        return None, None, None

def extract_year_ipc_vs_top10(filename):
    """
    Extracts year, IPC, and variant string (vs) from a filename.

    Expected filename format:
        'YYYY_IPC_vsInfo_aVC.csv'
    where:
        - YYYY is a 4-digit year,
        - IPC is a 4-character alphanumeric IPC code,
        - vsInfo is an arbitrary variant string (can include underscores),
        - and the filename ends strictly with '_aVC.csv'.

    Parameters:
        filename (str): The filename to parse.

    Returns:
        tuple: (year, ipc, vs) if the format matches, otherwise (None, None, None).
               Prints a warning if the format is unexpected.
    """
    match = re.match(r'(\d{4})_([A-Za-z0-9]{4})_(.*)_aVC\.csv', filename)
    if match:
        year = match.group(1)
        ipc = match.group(2)
        vs = match.group(3)
        return year, ipc, vs
    else:
        print(f"Filename format unexpected: {filename}")
        return None, None, None

def parse_stopwords(sW_path):
    """
    Load stopword lists from all .txt files in a directory and add NLTK English stopwords.

    Each file should contain one stopword per line. Returns a dictionary 
    where keys are filenames (without extensions) and values are lists 
    of stopwords.

    Parameters:
        sW_path (str): Path to the folder containing .txt stopword files.

    Returns:
        dict: Dictionary of stopword lists keyed by filename, including 'nltk_en'.
    """
    from nltk.corpus import stopwords
    stopEN = stopwords.words('english')
    stopword_dict = {}

    for filename in os.listdir(sW_path):
        if filename.endswith('.txt'):
            label = os.path.splitext(filename)[0]  # Use filename without extension as key
            file_path = os.path.join(sW_path, filename)
            
            with open(file_path, 'r', encoding='utf-8') as file:
                # Read lines and strip newline characters and extra spaces
                stopwords = [line.strip() for line in file if line.strip()]
            
            stopword_dict[label] = stopwords
    stopword_dict["nltk_en"] = stopEN
    return stopword_dict





def cleanCSV(pathData, yearList=None, ipcList=None, datasets_to_process="all"):
    """
    Clean and lemmatize patent CSV files for specified years, IPC codes, and datasets.
    Requires tE, KS, ES subfolders in data/csv_clean
    
    Loads raw patent CSV datasets (`tE`, `KS`, and `ES`), applies cleaning and 
    lemmatization using the `vocab` module, and writes the processed files 
    to the output directory. Ensures that all datasets share matching 
    year/IPC combinations.

    Parameters:
        pathData (str): Path to data (folder containing compressedData, csv_clean, csv_raw, jsonData, vocab)
        yearList (list or None): List of years to process; if None, all years in data are used.
        ipcList (list or None): List of IPC codes to process; if None, all IPCs in data are used.
        vocab (instance of Vocab): Instance created with technet, lemmatized technet, and stopwords.
        datasets_to_process (str): Which datasets to process: "tE", "KS", "ES", or "all" (default: "all").

    Returns:
        None
    """
    pathVocab = os.path.join(pathData, "vocab", "vocab")
    technet = pd.read_csv(os.path.join(pathVocab, "clean_technet.csv"))
    tn_lemm = pd.read_csv(os.path.join(pathVocab, "lemmatized_technet.csv"))

    sW_path = os.path.join(pathVocab, "additional stopwords")
    sW_dict = parse_stopwords(sW_path)
    stopwords = list(set([item for sublist in sW_dict.values() for item in sublist]))
    vocab = Vocab(technet=technet, df_lemm=tn_lemm, stopwords=stopwords)

    pathRaw = os.path.join(pathData, "csv_raw")
    pathClean = os.path.join(pathData, "csv_clean")


    tE_names = get_file_names(os.path.join(pathRaw, "tE"))
    KS_names = get_file_names(os.path.join(pathRaw, "KS"))
    ES_names = get_file_names(os.path.join(pathRaw, "ES"))
    tE_set = set([extract_year_ipc(string) for string in tE_names])
    KS_set = set([extract_year_ipc(string) for string in KS_names])
    ES_set = set([extract_year_ipc(string) for string in ES_names])

    assert(tE_set == KS_set == ES_set)

    if yearList is not None and ipcList is not None:
        yearList = yearList
        ipcList = ipcList
    else:
        yearList = [year for year, ipc in list(tE_set)]
        ipcList = [ipc for year, ipc in list(tE_set)]

    for year in yearList:
        for ipc in ipcList:
            print(f"{year}")
            print(f"    {ipc}")

            if datasets_to_process in ["tE", "all"]:
                tE = pd.read_csv(os.path.join(pathRaw, f"tE/{year}_{ipc}_patents_toEval.csv"))
                tE_clean = vocab.lemmDF(vocab.cleanDF(tE, type="all"))
                tE_clean.to_csv(os.path.join(pathClean, "tE", f"{year}_{ipc}_tE_cleaned.csv"), index=False)

            if datasets_to_process in ["KS", "all"]:
                matching_files = glob.glob(os.path.join(pathRaw, f"KS/{year}_*_{ipc}_KS_raw.csv"))[0]
                KS = pd.read_csv(matching_files)
                KS_clean = vocab.lemmDF(vocab.cleanDF(KS, type="all"))
                KS_clean.to_csv(os.path.join(pathClean, "KS", f"{year}_{ipc}_KS_cleaned.csv"), index=False)

            if datasets_to_process in ["ES", "all"]:
                matching_files = glob.glob(os.path.join(pathRaw, f"ES/{year}_*_{ipc}_ES_raw.csv"))[0]
                ES = pd.read_csv(matching_files)
                ES_clean = vocab.lemmDF(vocab.cleanDF(ES, type="all"))
                ES_clean.to_csv(os.path.join(pathClean, "ES", f"{year}_{ipc}_ES_cleaned.csv"), index=False)

    return None

        



# On peut enlever
def count_word_frequency_patent(text):
    words = text.split()
    word_count = Counter(words)
    return word_count

def count_word_frequency_total(df_clean):
    # Initialize a Counter to hold the total word frequency across the entire column
    total_word_count = Counter()

    # Iterate through the rows of the selected columns and update the total word count
    for column in list(df_clean.columns):
        for text in df_clean[column]:
            total_word_count += count_word_frequency_patent(text)
    return total_word_count

