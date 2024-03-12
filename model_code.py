# COLIEEE 2024

# =========================================
# MODEL_CODE
# Code to train model and perform inferences
# =========================================


import random, logging, sys, pickle, json
import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from itertools import product
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

logging.basicConfig(level=logging.INFO, #.WARNING
                    format='%(message)s',
                    stream=sys.stdout)


fields = ['prop_max_cos_sim_sents',
       'prop_max_cos_sim_paras', 'prop_max_jaccard_sents',
       'prop_max_jaccard_paras', 'prop_max_overlap_sents',
       'prop_max_overlap_paras', 'entity_set_jaccard',
       'sentences_en_set_jaccard', 'same_case', 'tfidf_entity_string',
       'tfidf_sentences_en_string', 'quotes_any', 'check_year',
       'judge_pair_ratio', 'judge_match', 'max_overall',
       'bin_0.00_0.10', 'bin_0.10_0.20', 'bin_0.20_0.30', 'bin_0.30_0.40',
       'bin_0.40_0.50', 'bin_0.50_0.60', 'bin_0.60_0.70', 'bin_0.70_0.80',
       'bin_0.80_0.90', 'bin_0.90_1.00']

# Save as keras native format:
def save_model(model):
    filename = './evaluation/model.keras'
    model.save(filename)
    print("Saved model: ", filename)

# Save model df pairs by separating out the model itself, and saving as .keras:
def save_model_df_pairs(model_df_pairs):
    save_version = []
    for i, (model, df1, df2) in enumerate(model_df_pairs):
        # Save each model to a separate file
        model_filename = f'./evaluation/model_{i}.keras'
        model.save(model_filename)

        # Replace the model object with the filename in the tuple
        save_version.append((model_filename, df1, df2))

    # Serializing the modified object
    with open('./evaluation/model_df_pairs.pkl', 'wb') as file:
        pickle.dump(save_version, file)

# Load and reconsistute the model pairs:
def load_model_df_pairs():


    # Deserializing the object
    with open('./evaluation/model_df_pairs.pkl', 'rb') as file:
        loaded_object = pickle.load(file)

    # Reconstruct the original structure with loaded models
    reconstructed_models_and_data = []
    for model_filename, df1, df2 in loaded_object:
        model = load_model(model_filename)
        reconstructed_models_and_data.append((model, df1, df2))

    return reconstructed_models_and_data

# Save pairs as pickle:
def save_train_df(train_df):
    filename = './evaluation/train_df.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(train_df, f)
    print("Saved pickle: ", filename)


# Train neural net model
def build_train_model(train_df):

    pd.options.mode.chained_assignment = None

    # Scale positive class samples to ~ 1/100 ratio:
    positive_class = train_df[train_df['match'] == 1]
    scale_factor = len(train_df) / len(positive_class) / 100
    multiplied_positive_class = pd.concat([positive_class] * round(scale_factor), ignore_index=True)
    train_df = pd.concat([train_df, multiplied_positive_class]).reset_index(drop=True)

    # Step 2: Define and train the Neural Network model
    X_train = train_df[fields]
    y_train = train_df['match']

    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    @tf.keras.utils.register_keras_serializable()
    def f1_score(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')  # Cast y_true to float32
        y_pred = K.round(y_pred)  # Round y_pred to 0 or 1

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())

        return 2 * (precision * recall) / (precision + recall + K.epsilon())

    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy',f1_score])

    # Fit the model
    model.fit(X_train, y_train, epochs=15, batch_size=32)

    logging.info("Returned trained model.")

    return model, train_df


# Method to apply inference on dev set:
def inference_on_dev(model, dev_df, infer_type):

    # Make inferences:
    if infer_type == 1:
        results_df = infer_1(model, dev_df)
    elif infer_type == 2:
        results_df = infer_2(model, dev_df)

    # Compute precision, recall, and F1 score
    y_test = results_df['match']
    predictions = results_df['predicted_class']

    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    num_predicted_class_1 = results_df[results_df['predicted_class'] == 1].shape[0]
    unique_query_predicted_class_1 = results_df[results_df['predicted_class'] == 1]['query'].nunique()
    unique_query_only_no_class_1 = results_df[~results_df['query'].isin(results_df[results_df['predicted_class'] == 1]['query'])]['query'].nunique()
    unique_target_predicted_class_1 = results_df[results_df['predicted_class'] == 1]['target'].nunique()
    unique_target_only_no_class_1 = results_df[~results_df['target'].isin(results_df[results_df['predicted_class'] == 1]['target'])]['target'].nunique()
    average_query_class_1 = results_df[results_df['predicted_class'] == 1]['query'].value_counts().mean()

    stats = {
        "num_predicted_class_1": num_predicted_class_1,
        "unique_query_predicted_class_1": unique_query_predicted_class_1,
        "unique_query_no_class_1": unique_query_only_no_class_1,
        "unique_target_predicted_class_1": unique_target_predicted_class_1,
        "unique_target_no_class_1": unique_target_only_no_class_1,
        "average_query_class_1": average_query_class_1
    }

    for k in stats.keys():
        print(k, stats[k])

    print("Model performance on dev: ")
    print("Precision: %f" % precision)
    print("Recall   : %f" % recall)
    print("F1       : %f" % f1)

    return results_df, precision, recall, f1

# Inference on min number:
def infer_1(model, df):

    min_matches = 2

    X_test = df[fields]
    logging.info("Applying model to get probabilities:")
    probabilities = model.predict(X_test).flatten()
    df['probability'] = probabilities
    df['predicted_class'] = 0

    sorted_df = df.sort_values(by='probability', ascending=False)

    # Add the top min_matches by proability for each unique query:
    unique_queries = sorted_df['query'].unique()
    for q in tqdm(unique_queries,total=len(unique_queries),desc="Checking all query files"):
        file_df = df[df['query'] == q]
        top_indices = file_df['probability'].nlargest(min_matches).index
        df.loc[top_indices, 'predicted_class'] = 1

    # Add the top 1x target for each unique target:
    unique_targets = sorted_df['target'].unique()
    for t in tqdm(unique_targets,total=len(unique_targets),desc="Checking all target files"):
        file_df = df[df['target'] == t]
        top_indices = file_df['probability'].nlargest(1).index
        df.loc[top_indices, 'predicted_class'] = 1

    return df

# Infer on all > threshold, with top targets & queries:
def infer_2(model, df):

    threshold = 0.65

    X_test = df[fields]
    logging.info("Applying model to get probabilities:")
    probabilities = model.predict(X_test).flatten()
    df['probability'] = probabilities
    df['predicted_class'] = 0

    # assign all > threshold:

    for index, row in tqdm(df.iterrows(),total=len(df),desc="Add > p probs"):
        if row['probability'] > threshold:
            df.loc[index,'predicted_class'] = 1

    # Top targets:
    sorted_df = df.sort_values(by='probability', ascending=False)
    marked_target_files = set()
    for index, row in tqdm(sorted_df.iterrows(),total=len(sorted_df),desc="Assigning targets"):
        if row['target'] not in marked_target_files:
            df.at[index, 'predicted_class'] = 1
            marked_target_files.add(row['target'])

    # Top queries:
    marked_query_files = set()
    for index, row in tqdm(sorted_df.iterrows(),total=len(sorted_df),desc="Assigning queries"):
        if row['query'] not in marked_query_files:
            df.at[index, 'predicted_class'] = 1
            marked_query_files.add(row['query'])

    return df



# Evaluate performance of model on train set, using 4x k-fold validation:
def get_k_fold_model_dev_pairs(pairs):

    # Helper method to get k-fold list of query file numbers:
    def get_k_fold_lists(unique_queries):
        n = 4
        random.seed(42)
        random.shuffle(unique_queries)
        part_size = len(unique_queries) // n
        remainder = len(unique_queries) % n

        k_fold_lists = []
        start = 0
        for i in range(n):
            end = start + part_size + (1 if i < remainder else 0)
            dev_queries = unique_queries[start:end]
            train_queries = [item for item in unique_queries if item not in dev_queries]
            k_fold_lists.append((dev_queries, train_queries))
            start = end

        return k_fold_lists

    # Helper: to split pairs df into train and dev dfs, based on non-overlapping lists of query files
    def get_train_and_test_df(train_queries, dev_queries, pairs, labels):

        train_df = pairs[pairs['query'].isin(train_queries)]

        dev_targets = []
        for query in dev_queries:
            dev_targets.extend([t.rstrip('.txt') for t in labels[query + '.txt']])
        dev_targets.sort()
        dev_tuples = [(query, target) for query, target in product(dev_queries, dev_targets) if query != target]

        dev_df = pairs[pairs['tuple'].isin(dev_tuples)]

        return (train_df, dev_df)

    json_train_labels_path = './data/task1_train_labels_2024.json'
    with open(json_train_labels_path, 'r') as f:
        labels = json.load(f)

    model_df_pairs = []

    unique_queries = pairs[pairs['set']=='train']['query'].unique()

    k_fold_lists = get_k_fold_lists(unique_queries)

    for k_fold in tqdm(k_fold_lists):

        dev_queries = k_fold[0]
        train_queries = k_fold[1]

        print("Using train_queries: len %f starting %s" % (len(train_queries), train_queries[0:4]))
        print("Using dev_queries  : len %f starting %s" % (len(dev_queries), dev_queries[0:4]))

        train_df, dev_df = get_train_and_test_df(train_queries, dev_queries, pairs, labels)

        # Applying judge ratio based on the train_df ONLY:
        training_pairs = train_df[(train_df['match']==1)&(train_df['set']=='train')]
        tqdm.pandas(desc="Judge matching based on k-fold train set only")
        logging.info("Judge matching based on k-fold train set only")
        value_counts_combined = training_pairs['judge_pair'].value_counts(normalize=True)
        train_df['judge_pair_ratio'] = train_df['judge_pair'].map(value_counts_combined).fillna(value_counts_combined.mean())
        train_df['judge_pair_ratio'] = train_df['judge_pair_ratio'] * 100
        dev_df['judge_pair_ratio'] = dev_df['judge_pair'].map(value_counts_combined).fillna(value_counts_combined.mean())
        dev_df['judge_pair_ratio'] = dev_df['judge_pair_ratio'] * 100

        model, train_df = build_train_model(train_df)
        model_df_pairs.append((model,dev_df,train_df))

    return model_df_pairs

def apply_models_to_dfs(model_df_pairs, infer_type):

    # Helper method to return means of results:
    def mean_of_tuple_positions(tuples):

        # Summing up the values in each position
        sum_pos = [0, 0, 0]
        for t in tuples:
            sum_pos[0] += t[0]
            sum_pos[1] += t[1]
            sum_pos[2] += t[2]

        # Calculating the mean for each position
        mean_pos = (sum_pos[0] / len(tuples), sum_pos[1] / len(tuples), sum_pos[2] / len(tuples))
        return mean_pos

    results = []

    logging.info("Using infer type: %d" % infer_type)

    for i, (model,dev_df, train_df) in enumerate(model_df_pairs):

        print("df: %d" % i)
        results_df, precision, recall, f1 = inference_on_dev(model,dev_df, infer_type)
        results.append((precision, recall, f1))
        print("-------------------------------------")

    print()
    print("==RESULTS=============================")
    for i, r in enumerate(results):
        print(r)
    print("Means:")
    print(mean_of_tuple_positions(results))


# Method to apply inference on test set and export results:
def inference_on_test(model, test_df, infer_type):

    # Make inferences:
    if infer_type == 1:
        results_df = infer_1(model, test_df)
    elif infer_type == 2:
        results_df = infer_2(model, test_df)

    # stats
    num_predicted_class_1 = results_df[results_df['predicted_class'] == 1].shape[0]
    unique_query_predicted_class_1 = results_df[results_df['predicted_class'] == 1]['query'].nunique()
    unique_query_only_no_class_1 = results_df[~results_df['query'].isin(results_df[results_df['predicted_class'] == 1]['query'])]['query'].nunique()
    unique_target_predicted_class_1 = results_df[results_df['predicted_class'] == 1]['target'].nunique()
    unique_target_only_no_class_1 = results_df[~results_df['target'].isin(results_df[results_df['predicted_class'] == 1]['target'])]['target'].nunique()
    average_query_class_1 = results_df[results_df['predicted_class'] == 1]['query'].value_counts().mean()

    stats = {
        "num_predicted_class_1": num_predicted_class_1,
        "unique_query_predicted_class_1": unique_query_predicted_class_1,
        "unique_query_no_class_1": unique_query_only_no_class_1,
        "unique_target_predicted_class_1": unique_target_predicted_class_1,
        "unique_target_no_class_1": unique_target_only_no_class_1,
        "average_query_class_1": average_query_class_1
    }
    print(stats)

    # Export results
    filtered_df = results_df[results_df['predicted_class'] == 1].sort_values(by=['query', 'target'])
    formatted_lines = filtered_df.apply(lambda row: f"{row['query']} {row['target']} UMNLP{infer_type}", axis=1)

    # Exporting to a .txt file
    filename = f'./output/results_{infer_type}.txt'
    with open(filename, 'w') as file:
        for line in formatted_lines:
            file.write(line + '\n')

    logging.info("Written results to file %s" % filename)

    # Validate file:----------------
    # Initialize sets for unique first and second position strings
    unique_first_strings = set()
    unique_second_strings = set()

    # Initialize a counter for the total number of lines
    total_lines = 0

    # Open and read the file
    with open(filename, 'r') as file:
        for line in file:
            total_lines += 1  # Increment the line count

            # Split each line into parts
            parts = line.split()

            # Add the first and second strings to their respective sets
            if len(parts) >= 2:  # Check if there are at least two parts
                unique_first_strings.add(parts[0])
                unique_second_strings.add(parts[1])

    # Calculate the number of unique strings in each position
    num_unique_first = len(unique_first_strings)
    num_unique_second = len(unique_second_strings)

    # Calculate the average number of lines per unique first position
    average_lines_per_unique_first = total_lines / num_unique_first

    # Print the results
    print("Total number of lines:", total_lines)
    print("Number of unique strings in the first position:", num_unique_first)
    print("Number of unique strings in the second position:", num_unique_second)
    print("Average number of lines per unique first position:", average_lines_per_unique_first)
    # END validation ------------


    return results_df
