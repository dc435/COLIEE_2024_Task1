# COLIEEE 2024

# =========================================
# PAIRS_CODE
# Code to generate fields for 'pairs' dataframe (ie. comparison of paired case features)
# =========================================


import logging, sys, pickle, re, json, os
from itertools import product
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, #.WARNING
                    format='%(message)s',
                    stream=sys.stdout)




# Save pairs as pickle:
def save_pairs(pairs):
    filename = './files/pairs.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(pairs, f)
    print("Saved pickle: ", filename)

# Load pairs from pickle:
def load_pairs():
    filename = './files/pairs.pkl'
    with open(filename, 'rb') as f:
        pairs = pickle.load(f)
    return pairs


# Method to build original pair dataframe:
def get_pairs(files):

    logging.info('Generating pairs dataframe from files.')

    # Train:
    json_train_labels_path = './data/task1_train_labels_2024.json'
    with open(json_train_labels_path, 'r') as f:
        train_labels = json.load(f)
    train_queries = [t.rstrip('.txt') for t in list(train_labels.keys())]
    train_targets = []
    for file in train_queries:
        train_targets.extend([t.rstrip('.txt') for t in train_labels[file + '.txt']])
    train_queries.sort()
    train_targets = list(set(train_targets))
    train_targets.sort()
    train_tuples = [(query, target) for query, target in product(train_queries, train_targets) if query != target]
    train_df = pd.DataFrame([{'query': query, 'target': target, 'tuple': (query, target), 'set': 'train'} for query, target in train_tuples])

    # Test:
    json_test_labels_path = './data/task1_test_no_labels_2024.json'
    with open(json_test_labels_path, 'r') as f:
        test_queries = [t.rstrip('.txt') for t in json.load(f)]
    test_targets = []
    for filename in files[files['set']=='test']['filename']:
        if filename not in test_queries:
            test_targets.append(filename)
    test_queries.sort()
    test_targets.sort()
    test_tuples = [(query, target) for query, target in product(test_queries, test_targets) if query != target]
    test_df = pd.DataFrame([{'query': query, 'target': target, 'tuple': (query, target), 'set': 'test'} for query, target in test_tuples])

    pairs = pd.concat([train_df, test_df], ignore_index=True)

    # Add case matches:
    cases_dict = files.set_index('filename')['cases'].to_dict()

    def get_match(row):
        if row['target'] + '.txt' in cases_dict[row['query']]:
            return int(1)
        elif row['set'] == 'test':
            return np.nan
        else:
            return int(0)

    tqdm.pandas(desc="Getting matches")
    pairs['match'] = pairs.progress_apply(get_match, axis=1)

    return pairs


# =============================================================================================
# BINS:

# Method to get similarity score percentages as histogram in bins from 0 to 1
def add_bins(files, pairs):

    logging.info('Generating histogram bins using cosine similarity from sentence_en embeddings.')

    # Custom bins
    bins = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

    # Add new columns for histogram bins based on the custom bins
    bin_labels = []
    for i in range(len(bins) - 1):
        bin_labels.append(f'bin_{bins[i]:.2f}_{bins[i+1]:.2f}')
        pairs[bin_labels[i]] = 0.0

    embeddings_dict = files.set_index('filename')['embeddings_sentences_en'].to_dict()

    def update_row(row):

        # Wrap in try block in case zero embeddings for either query or target
        try:
            query_matrix = np.stack(embeddings_dict[row['query']])
            target_matrix = np.stack(embeddings_dict[row['target']])
        except:
            return [0] * (len(bins) - 1)
        similarity_matrix = cosine_similarity(query_matrix, target_matrix).flatten()
        hist, _ = np.histogram(similarity_matrix, bins=bins, density=False)
        total_counts = np.sum(hist)
        hist_percentage = hist / total_counts if total_counts > 0 else hist
        return pd.Series(hist_percentage)

    tqdm.pandas(desc="Getting histogram bins")
    pairs[bin_labels] = pairs.progress_apply(update_row, axis=1)

    logging.info('Histogram bins added to "pairs" df.')

# Method to add scalar of difference in size of target and query cases
def add_scalars(files, pairs):

    sent_para_dict = files.set_index('filename')['sentences_en'].to_dict()

    def add_scalar(row):
        if len(sent_para_dict[row['target']]) > 0:
            return len(sent_para_dict[row['query']]) / len(sent_para_dict[row['target']])
        else:
            return 0

    tqdm.pandas(desc="Adding scalars")
    pairs['scalar'] = pairs.progress_apply(add_scalar, axis=1)

    logging.info('Added scalar to pairs.')



# =============================================================================================
# PROPOSITIONS:


# Get maximum cosine similarity between the query propositions and the target sents_en, for each pair:
def get_prop_max_cos_sim_sents(files, pairs):

    logging.info('Getting max cos similarity between query propositions and target sentences_en.')

    query_embeddings_dict = files.set_index('filename')['embeddings_propositions_en'].to_dict()
    target_embeddings_dict = files.set_index('filename')['embeddings_sentences_en'].to_dict()

    def get_max(row):
        try:
            query_matrix = np.stack(query_embeddings_dict[row['query']])
            target_matrix = np.stack(target_embeddings_dict[row['target']])
        except:
            return 0
        similarity_matrix = cosine_similarity(query_matrix, target_matrix)
        return similarity_matrix.max()

    tqdm.pandas(desc="Getting max cosine similarity")
    pairs['prop_max_cos_sim_sents'] = pairs.progress_apply(get_max, axis=1)

    logging.info('Maximum proposition cosine similarity (with sentences_en) added to pairs.')


# Get maximum cosine similarity between the query propositions and the target paras, for each pair:
def get_prop_max_cos_sim_paras(files, pairs):

    logging.info('Getting max cos similarity between query propositions and target paragraphs_formatted.')

    query_embeddings_dict = files.set_index('filename')['embeddings_propositions_en'].to_dict()
    target_embeddings_dict = files.set_index('filename')['embeddings_paragraphs_formatted'].to_dict()

    def get_max(row):
        try:
            query_matrix = np.stack(query_embeddings_dict[row['query']])
            target_matrix = np.stack(target_embeddings_dict[row['target']])
        except:
            return 0
        similarity_matrix = cosine_similarity(query_matrix, target_matrix)
        return similarity_matrix.max()

    tqdm.pandas(desc="Getting max cosine similarity")
    pairs['prop_max_cos_sim_paras'] = pairs.progress_apply(get_max, axis=1)

    logging.info('Maximum proposition cosine similarity (with paragraphs_formatted) added to pairs.')


# Helper function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if not union:
        return 0  # Avoid division by zero
    return len(intersection) / len(union)

# Get maximum jaccard sim between proposition word set - sentences word sets:
def get_prop_max_jaccard_sents(files, pairs):

    logging.info('Getting max jaccard similarity between sentences_en_sets and propositions_en_sets.')

    sentences_en_set_dict = files.set_index('filename')['sentences_en_set_list'].to_dict()
    propositions_en_set_dict = files.set_index('filename')['propositions_en_set_list'].to_dict()

    def get_max(row):

        query_sets = propositions_en_set_dict[row['query']]
        target_sets = sentences_en_set_dict[row['target']]

        max_jaccard = 0
        for qs in query_sets:
            for ts in target_sets:
                jaccard = jaccard_similarity(ts,qs)
                max_jaccard = max(max_jaccard, jaccard)

        return max_jaccard

    tqdm.pandas(desc="Getting max jaccard proposition-sentences")
    pairs['prop_max_jaccard_sents'] = pairs.progress_apply(get_max, axis=1)

    logging.info('Added prop_max_jaccard_sents to pairs.')

# Get maximum jaccard sim between proposition word set - paragraphs word sets:
def get_prop_max_jaccard_paras(files, pairs):

    logging.info('Getting max jaccard similarity between sentences_en_sets and paragraphs_formatted_set.')

    paragraphs_formatted_set_dict = files.set_index('filename')['paragraphs_formatted_en_set_list'].to_dict()
    propositions_en_set_dict = files.set_index('filename')['propositions_en_set_list'].to_dict()

    def get_max(row):

        query_sets = propositions_en_set_dict[row['query']]
        target_sets = paragraphs_formatted_set_dict[row['target']]

        max_jaccard = 0
        for qs in query_sets:
            for ts in target_sets:
                jaccard = jaccard_similarity(ts,qs)
                max_jaccard = max(max_jaccard, jaccard)

        return max_jaccard

    tqdm.pandas(desc="Getting max jaccard proposition-paragraphs")
    pairs['prop_max_jaccard_paras'] = pairs.progress_apply(get_max, axis=1)

    logging.info('Added prop_max_jaccard_paras to pairs.')

# Get maximum overlap ratio of proposition word set - sentences word sets:
def get_prop_max_overlap_sents(files, pairs):

    logging.info('Getting max overlap ratio between sentences_en_sets and propositions_en_sets.')

    sentences_en_set_dict = files.set_index('filename')['sentences_en_set_list'].to_dict()
    propositions_en_set_dict = files.set_index('filename')['propositions_en_set_list'].to_dict()

    def get_max(row):

        query_sets = propositions_en_set_dict[row['query']]
        target_sets = sentences_en_set_dict[row['target']]

        max_overlap = 0
        for qs in query_sets:
            if len(qs) > 5: #Ensure proposition set >5 to exclude short, common propositions
                for ts in target_sets:
                    overlap = len(qs.intersection(ts)) / len(qs)
                    max_overlap = max(max_overlap, overlap)

        return max_overlap

    tqdm.pandas(desc="Getting max overlap proposition-sentences")
    pairs['prop_max_overlap_sents'] = pairs.progress_apply(get_max, axis=1)

    logging.info('Added prop_max_overlap_sents to pairs.')

# Get maximum overlap ratio of proposition word set - paragraphs word sets:
def get_prop_max_overlap_paras(files, pairs):

    logging.info('Getting max overlap ratio between paragraphs_formatted_set and propositions_en_sets.')

    paragraphs_formatted_set_dict = files.set_index('filename')['paragraphs_formatted_en_set_list'].to_dict()
    propositions_en_set_dict = files.set_index('filename')['propositions_en_set_list'].to_dict()

    def get_max(row):

        query_sets = propositions_en_set_dict[row['query']]
        target_sets = paragraphs_formatted_set_dict[row['target']]

        max_overlap = 0
        for qs in query_sets:
            if len(qs) > 5: #Ensure proposition set >5 to exclude short, common propositions
                for ts in target_sets:
                    overlap = len(qs.intersection(ts)) / len(qs)
                    max_overlap = max(max_overlap, overlap)

        return max_overlap

    tqdm.pandas(desc="Getting max overlap proposition-paragraphs")
    pairs['prop_max_overlap_paras'] = pairs.progress_apply(get_max, axis=1)

    logging.info('Added prop_max_overlap_paras to pairs.')


# =============================================================================================
# CASE-CASE:

# Add jaccard similarities between cases based on entity_sets and word_sets:
def get_case_jaccard_sims(files, pairs):

    logging.info('Getting jaccard similarity score between cases for both entity and word set.')

    entity_set_dict = files.set_index('filename')['entity_set'].to_dict()
    word_set_dict = files.set_index('filename')['sentences_en_set'].to_dict()

    # Calculate Jaccard similarity for each row in pairs
    tqdm.pandas(desc="Calculating entity set jaccard similarity")
    pairs['entity_set_jaccard'] = pairs.progress_apply(lambda row: jaccard_similarity(entity_set_dict[row['query']], entity_set_dict[row['target']]), axis=1)
    tqdm.pandas(desc="Calculating sentences en set jaccard similarity")
    pairs['sentences_en_set_jaccard'] = pairs.progress_apply(lambda row: jaccard_similarity(word_set_dict[row['query']], word_set_dict[row['target']]), axis=1)

    logging.info('Added jaccard similarity scores in entity_set_jaccard and sentences_en_set_jaccard.')


# Method to apply binary value to cases which appear to be identical:
def check_same_case(pairs):

    def check_same(row):
        jacc_ent_set_score = row['entity_set_jaccard']
        jacc_word_set_score = row['sentences_en_set_jaccard']
        if jacc_ent_set_score > 0.90 and jacc_word_set_score > 0.90:
            return 1
        else:
            return 0

    pairs['same_case'] = pairs.progress_apply(check_same, axis=1)

    logging.info('Add same_case check to pairs.')


# Method to calculate tfidf scores:
def get_case_tfidf_scores(files, pairs):

    def tfidf_calculator(string_field):

        logging.info('Getting tfidf for ' + string_field)

        train_files = files[files['set']=='train']
        test_files = files[files['set']=='test']

        # Initialize TFIDF Vectorizer and fit on training data
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vectorizer.fit(train_files[string_field])

        # Transform both train and test data
        train_tfidf_matrix = tfidf_vectorizer.transform(train_files[string_field])
        test_tfidf_matrix = tfidf_vectorizer.transform(test_files[string_field])

        # Create dictionaries for train and test TF-IDF vectors
        vector_dictionary = dict(zip(train_files['filename'], train_tfidf_matrix))
        vector_dictionary.update(dict(zip(test_files['filename'], test_tfidf_matrix)))

        # Function to calculate cosine similarity
        def calculate_similarity(row):
            query_vector = vector_dictionary[row['query']]
            target_vector = vector_dictionary[row['target']]
            return cosine_similarity(query_vector, target_vector)[0][0]

        # TFIDF entity string calculations
        description = "Calculating tfidf for " + string_field
        tqdm.pandas(desc=description)
        new_field = "tfidf_" + string_field

        # Applying similarity calculation
        pairs[new_field] = pairs.progress_apply(calculate_similarity, axis=1)

    tfidf_calculator('entity_string')
    tfidf_calculator('sentences_en_string')

    logging.info('Added tfidf scores to pairs.')


# Number of query quotes in target text:
def get_num_quotes(files, pairs):

    logging.info('Get number of query quotes in target text.')

    quotes_dict = files.set_index('filename')['quotes'].to_dict()
    text_dict = files.set_index('filename')['text'].to_dict()

    # Helper method to improve quote matching:
    def clean_text(text):
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Convert to lower case
        text = text.lower()
        # Remove new lines
        text = text.replace('\n', ' ')
        # Remove digits
        text = re.sub(r'\d+', '', text)
        text = text.strip()
        # Remove any space greater than 1
        text = re.sub(r'\s+', ' ', text)

        return text

    def check_quotes_row(row):

        quotes = quotes_dict[row['query']]
        # Truncate quotes to first 10 words to aid quote matching (in case of slight typo)
        truncated_quotes = [' '.join(clean_text(quote).split()[:10]) for quote in quotes]
        text = clean_text(text_dict[row['target']])

        num_quotes = 0
        for q in truncated_quotes:
            if q in text:
                num_quotes += 1

        return num_quotes

    tqdm.pandas(desc="Getting number of query quotes")
    pairs['quotes_num'] = pairs.progress_apply(check_quotes_row, axis=1)


def binarize_quotes(pairs):

    def check_any_quotes(number):
        if number > 0:
            return 1
        else:
            return 0
    pairs['quotes_any'] = pairs['quotes_num'].progress_apply(check_any_quotes)

    logging.info('Added quotes to pairs.')


# Method to check if target case pre-dates query case
def check_years(files,pairs):

    logging.info('Checking if target case pre-dates query case year.')

    year_dict = files.set_index('filename')['year'].to_dict()

    # Return 1 if query is later (or equal) to target year, 0 if not.
    def check_year(row):
        return int(year_dict[row['query']] >= year_dict[row['target']])

    tqdm.pandas(desc="Checking years")
    pairs['check_year'] = pairs.progress_apply(check_year, axis=1)

    logging.info('Added check_year to pairs.')


# Method to add the ratio of (query_judge,pair_judge) tuples in the matches
# Also add judge_match if judges match (1 for match, 0 otherwise)
def add_judge_checks(files, pairs):

    judge_dict = files.set_index('filename')['judge'].to_dict()

    def get_judget_pair(row):
        return (judge_dict[row['query']],judge_dict[row['target']])

    tqdm.pandas(desc="Add judge pairs")
    pairs['judge_pair'] = pairs.progress_apply(get_judget_pair, axis=1)

    training_pairs = pairs[(pairs['match']==1)&(pairs['set']=='train')]

    tqdm.pandas(desc="Judge matching")
    value_counts_combined = training_pairs['judge_pair'].value_counts(normalize=True)
    pairs['judge_pair_ratio'] = pairs['judge_pair'].map(value_counts_combined).fillna(value_counts_combined.mean())
    pairs['judge_pair_ratio'] = pairs['judge_pair_ratio'] * 100

    logging.info('Added judge_pair_ratio to pairs.')

    def check_judge_match(pair):
        return 1 if pair[0] == pair[1] else 0

    pairs['judge_match'] = pairs['judge_pair'].apply(check_judge_match)

    logging.info('Added judge_match to pairs.')


# Add maximum proposition cos-sim for any sentence-propositions combo:
def add_max_overall(pairs, files):
    logging.info('Adding maximum overall proposition matching scores.')

    propositions_embeddings_dict = files.set_index('filename')['embeddings_propositions_en'].to_dict()
    sentence_embeddings_dict = files.set_index('filename')['embeddings_sentences_en'].to_dict()

    grouped_pairs = pairs.groupby('query')
    pairs['max_overall'] = 0

    for uq, group in tqdm(grouped_pairs, desc="Adding max prop-sent cossim overall"):
        proposition_embeddings = propositions_embeddings_dict[uq]
        unique_targets = group['target'].unique()

        # Prepare target matrices and indices
        target_matrices = []
        target_indices = []

        for target in unique_targets:
            try:
                target_matrix = np.stack(sentence_embeddings_dict[target])
                target_matrices.append(target_matrix)
                target_indices.extend([group[group['target'] == target].index[0]] * len(target_matrix))
            except:
                continue

        if not target_matrices:
            continue

        combined_target_matrix = np.vstack(target_matrices)

        for p in proposition_embeddings:
            try:
                query_matrix = np.stack([p])
            except:
                continue

            cossim_matrix = cosine_similarity(query_matrix, combined_target_matrix)
            max_index = cossim_matrix.argmax()
            max_score = cossim_matrix.max()

            if max_score > 0:
                pairs.at[target_indices[max_index], 'max_overall'] = max_score

    logging.info('Added max_overall to pairs.')
