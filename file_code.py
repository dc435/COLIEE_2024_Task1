# COLIEEE 2024

# =========================================
# FILE_CODE
# Code to generate fields for 'files' dataframe (ie. the individual case features)
# =========================================


import json, os, re, logging, sys, torch, gc, warnings, pickle, spacy, string
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoModel
from torch.utils.data import DataLoader, Dataset
from nltk.stem import PorterStemmer

warnings.filterwarnings('ignore', category=UserWarning, module='tqdm')
pd.options.mode.chained_assignment = None
logging.basicConfig(level=logging.INFO,
                    format='%(message)s',
                    stream=sys.stdout)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()


# Save files as pickle:
def save_files(files):
    filename = './files/files.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(files, f)
    print("Saved pickle: ", filename)

# Load files from pickle:
def load_files():
    filename = './files/files.pkl'
    with open(filename, 'rb') as f:
        files = pickle.load(f)
    return files


# Read-in text files
# Return: 'files' dataframe
def get_files():

    logging.info('Reading raw data from text files. Generating files dataframe.')

    json_train_labels_path = './data/task1_train_labels_2024.json'
    json_test_labels_path = './data/task1_test_no_labels_2024.json'
    train_files_path = './data/task1_train_files_2024/'
    test_files_path = './data/task1_test_files_2024/'

    # Get train & dev file lists:
    with open(json_train_labels_path, 'r') as f:
        train_labels = json.load(f)
    train_queries = [t.rstrip('.txt') for t in list(train_labels.keys())]
    train_targets = []
    for file in train_queries:
        train_targets.extend([t.rstrip('.txt') for t in train_labels[file + '.txt']])

    # Read-in train & dev files:
    files = pd.DataFrame(columns=['filename','set','query','cases','text'])

    for file in tqdm(os.listdir(train_files_path),desc="Read-in train files"):
        filename = file.rstrip('.txt')
        if any(filename in filelist for filelist in [train_queries, train_targets]):
            file_path = os.path.join(train_files_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                temp_df = pd.DataFrame({
                    'filename': [filename],
                    'set': ['train'],
                    'query':[filename in train_queries],
                    'cases': [train_labels[file] if file in train_labels else []],
                    'text': [text]})
                files = pd.concat([files, temp_df], ignore_index=True)

    # Get test file lists:
    with open(json_test_labels_path, 'r') as f:
        test_queries = [t.rstrip('.txt') for t in json.load(f)]
    test_targets = []
    for file in os.listdir(test_files_path):
        if file.rstrip('.txt') not in test_queries:
            test_targets.append(file.rstrip('.txt'))

    # Read-in test files:
    for file in tqdm(os.listdir(test_files_path),desc="Read-in test files"):
        filename = file.rstrip('.txt')
        file_path = os.path.join(test_files_path, file)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            temp_df = pd.DataFrame({
                'filename': [filename],
                'set': ['test'],
                'query':[filename in test_queries],
                'cases': [[]],
                'text': [text]})
            files = pd.concat([files, temp_df], ignore_index=True)

    logging.info('Returning "files" df.')

    return files



# Method to extract paragraphs from raw text, and add to files df
def add_paragraphs(files):

    logging.info('Reading text from files. Extracting paragraphs based on regex pattern.')

    def get_paragraphs(text):

        pattern = r'(\[\d{1,4}\])'
        paragraphs = []
        text = text.replace('\n', ' ').replace('•',' ')
        text = re.sub(r'\s+', ' ', text)

        # Use regex to split the raw text into paragraphs based on [1], [2] ... formatting:

        start_indices = [match.start() for match in re.finditer(pattern, text)]

        if start_indices:
            rolling_count = 1

            for i in range(len(start_indices)):
                start = start_indices[i]
                target_text = text[start:start+6]
                match = re.search(pattern, target_text)
                if match:
                    number = int(match.group().strip('[]').strip('.'))
                else:
                    continue

                end = start_indices[i + 1] if i + 1 < len(start_indices) else len(text)

                # check if the bracketed number is within buffer range of the previous number:
                check_no = number - rolling_count
                if check_no <= 6 and check_no >= 0:
                    paragraphs.append(text[start:end])
                    rolling_count = number + 1
                elif len(paragraphs) > 0:
                    paragraphs[-1] += text[start:end]

        else:
            # If no matches, return the entire text
            paragraphs.append(text)

        return paragraphs

    tqdm.pandas(desc="Extracting all paragraphs")
    files['paragraphs'] = files['text'].progress_apply(get_paragraphs)

    logging.info('Updated "files" df has with "paragraphs".')


# Helper function to clean text (for get_sentences and ors.):
def clean_text(text):

    pattern1 = "FRAGMENT_SUPPRESSED"
    pattern2 = "REFERENCE_SUPPRESSED"
    pattern3 = "CITATION_SUPPRESSED"

    combined_pattern = f"{pattern1}|{pattern2}|{pattern3}"
    numbering_pattern = r'\[\d+\]'
    angle_pattern = r'[<>]'
    end_pattern = r"\[End of document\]"

    text = text.replace('\n', ' ').replace('•',' ')
    text = re.sub(combined_pattern,' ', text)
    text = re.sub(numbering_pattern,' ', text)
    text = re.sub(angle_pattern,' ', text)
    text = re.sub(end_pattern,' ', text)
    text = re.sub(r'\s+', ' ', text)

    text = text.strip()

    return text


# Method to extract formatted paragraphs
def get_paragraphs_formatted(files):

    logging.info('Getting formatted paragraphs of length < 250 words.')

    def extract_paragraphs_formatted(row):

        # All paragraphs shorter than 250 words:
        paragraphs_formatted = list([clean_text(p) for p in row['paragraphs'] if len(p.split()) < 250])
        paragraphs_formatted = [p for p in paragraphs_formatted if len(p.split()) > 1]

        return paragraphs_formatted

    tqdm.pandas(desc="Getting formatted paragraphs")
    files['paragraphs_formatted'] = files.progress_apply(extract_paragraphs_formatted, axis=1)

    logging.info('Added formatted paragraphs to "files" df in "paragraphs_formatted".')


# Extract suppressed section from paragraphs, using regex and some spacy
def add_suppressed_sections(files):

    logging.info('Using regex and spacy (for long paragraphs) to extract and modify suppressed sections from paragraphs:')

    # Helper method for get_suppressed_sections
    # Only called when the word count of the paragraph is longer than an upper limit
    # It takes long paragraph and returns a smaller paragraph centred around the marker
    def get_reduced_paragraph(paragraph, upper_limit, lower_limit, marker):

        doc = nlp(paragraph)
        sentences = [sent.text for sent in doc.sents]

        # Find the sentence with the marker
        target_index = None
        for i, sentence in enumerate(sentences):
            if marker in sentence:
                target_index = i
                break

        if target_index is None:
            return paragraph

        # Initialize the result with the target sentence
        initial_sentence = sentences[target_index]
        words_in_sentence = initial_sentence.split()

        if len(words_in_sentence) > upper_limit:
            # Convert character index to word index
            char_index = initial_sentence.find(marker)
            marker_word_index = len(initial_sentence[:char_index].split())

            # Calculate the start and end indices for the 250-word subsection
            start_index = max(0, marker_word_index - 125)
            end_index = min(len(words_in_sentence), start_index + 250)

            centered_subsection = ' '.join(words_in_sentence[start_index:end_index])
            result = [centered_subsection]
        else:
            result = [initial_sentence]

        word_count = len(result[0].split())

        # Expand before and after the marker sentence
        before_index, after_index = target_index - 1, target_index + 1
        while word_count < lower_limit and word_count < upper_limit:
            added = False

            # Add sentence from before
            if before_index >= 0:
                before_words = sentences[before_index].split()
                if word_count + len(before_words) <= upper_limit:
                    result.insert(0, sentences[before_index])
                    word_count += len(before_words)
                else:
                    # Add partial sentence from the end
                    needed_words = upper_limit - word_count
                    partial_sentence = ' '.join(before_words[-needed_words:])
                    result.insert(0, partial_sentence)
                    word_count = upper_limit
                added = True
                before_index -= 1

            if word_count >= lower_limit or word_count >= upper_limit:
                break

            # Add sentence from after
            if after_index < len(sentences) and not added:
                after_words = sentences[after_index].split()
                if word_count + len(after_words) <= upper_limit:
                    result.append(sentences[after_index])
                    word_count += len(after_words)
                else:
                    # Add partial sentence from the start
                    needed_words = upper_limit - word_count
                    partial_sentence = ' '.join(after_words[:needed_words])
                    result.append(partial_sentence)
                    word_count = upper_limit
                after_index += 1

        return ' '.join(result)

    # Helper method
    # Get Suppressed Sections. Identifies those with 'FRAGMENT_SUPPRESSED' and similar. Replaces fragment with a marker
    def get_suppressed_sections(row):

        # Check if the row relates to a query. If not, can return empty list (as the suppressions are not used in target cases):
        if not (row['query']):
            return []

        paragraphs = row['paragraphs']

        pattern1 = "FRAGMENT_SUPPRESSED"
        pattern2 = "REFERENCE_SUPPRESSED"
        pattern3 = "CITATION_SUPPRESSED"

        combined_pattern = f"{pattern1}|{pattern2}|{pattern3}"

        replacement_pattern = 'REFERENCE'
        marker = "TARGETCASE"
        lower_limit = 150
        upper_limit = 250

        supressed_sections = []

        for p in paragraphs:

            modified_paragraph = re.sub(combined_pattern, replacement_pattern, p)
            matches = re.finditer(replacement_pattern, modified_paragraph)

            for match in matches:
                marked_paragraph = modified_paragraph[:match.start()] + marker + modified_paragraph[match.end():]
                if len(marked_paragraph.split()) > upper_limit:
                    supressed_sections.append(get_reduced_paragraph(marked_paragraph, upper_limit, lower_limit, marker))
                else:
                    supressed_sections.append(marked_paragraph)

        return supressed_sections

    tqdm.pandas(desc="Extracting suppressed sections from paragraphs")
    files['suppressed_sections'] = files.progress_apply(get_suppressed_sections, axis=1)

    logging.info('Added suppressed sections to "suppressed_sections" field.')

    # No return



# Method to add lists of proposition strings to query files, using t5 transformer with suppressed sections
def add_propositions(files):

    logging.info('Using pre-trained t5 model to extract propositions from suppressed sections:')

    def get_propositions(row, tokenizer, model):

        # Check if the row relates to a query. If not, can return empty list (as the propositions are not used in target cases):
        if not (row['query']):
            return []

        def chunk_list(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]

        input_texts = row['suppressed_sections']

        # Split the list into batches of size 16
        batched_suppressions = list(chunk_list(input_texts, 32))

        all_decoded_outputs = []
        for batch in batched_suppressions:
            prefix = 'question: what is the proposition in TARGETCASE? context: '
            prefixed_suppressions = [prefix + s for s in batch]
            input_texts_tokenized = tokenizer(prefixed_suppressions, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device).input_ids
            outputs_tokenized = model.generate(input_texts_tokenized, max_length=128)
            decoded_outputs = tokenizer.batch_decode(outputs_tokenized, skip_special_tokens=True)

            # Append the decoded outputs of the current batch to the combined list
            all_decoded_outputs.extend(decoded_outputs)

        # Convert to set to deduplicate:
        return list(set(all_decoded_outputs))

    path='./models/trained_t5/'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=path)
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=path).to(device)

    tqdm.pandas(desc="Generating propositions from suppressed sections")
    files['propositions'] = files.progress_apply(lambda row: get_propositions(row, tokenizer, model), axis=1)

    logging.info('Added propositions to "propositions" field.')




# Helper method:
def get_english_sections(sections, tokenizer, model, device):

    gc.collect()
    torch.cuda.empty_cache()

    if len(sections) == 0:
        return sections

    # Function to process a batch of sections
    def process_batch(batch_sections):
        inputs = tokenizer(batch_sections, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class_ids = logits.argmax(dim=-1)
            return [p for p, class_id in zip(batch_sections, predicted_class_ids) if model.config.id2label[class_id.item()] == 'en']

    # Batch processing if more than 100 sections
    if len(sections) > 100:
        batch_size = 100
        filtered_sections = []

        for i in range(0, len(sections), batch_size):
            batch = sections[i:i + batch_size]
            filtered_sections.extend(process_batch(batch))
    else:
        filtered_sections = process_batch(sections)

    return filtered_sections

# Method to filter out non-English paragraphs
def get_english_propositions(files):

    logging.info('Using language detection model to filter non-English propositions.')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="papluca/xlm-roberta-base-language-detection")
    model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection").to(device)

    tqdm.pandas(desc="Getting English propositions")
    files['propositions_en'] = files['propositions'].progress_apply(lambda sections: get_english_sections(sections, tokenizer, model, device))

    logging.info('Added English-only propositions to "files" df in "propositions_en".')


# Method to add sentences to df, from the English paragraphs
def add_sentences(files):

    logging.info('Using spacy to extract sentences of char length > 25 from paragraphs:')

    # In: List of paragraphs
    # Return: List of sentences with char length > 25 (to exclude any single word / very common sentences)
    def get_sentences(paragraphs):

        text = ' '.join(paragraphs)
        sentences = []
        text = clean_text(text)
        doc = nlp(text)
        for s in [sent.text for sent in doc.sents]:
            if len(s) > 25:
                sentences.append(s)
        return sentences

    tqdm.pandas(desc="Extracting sentences from paragraphs")
    files['sentences'] = files['paragraphs'].progress_apply(get_sentences)

    logging.info('Added lists of sentences to "sentences".')


# Method to filter out non-English sentences
def get_english_sentences(files):

    logging.info('Using language detection model to filter non-English sentences.')

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="papluca/xlm-roberta-base-language-detection")
    model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection").to(device)

    tqdm.pandas(desc="Getting English sentences")
    files['sentences_en'] = files['sentences'].progress_apply(lambda sections: get_english_sections(sections, tokenizer, model, device))

    logging.info('Added English-only sentences to "files" df in "sentences_en".')


# Method to extract quotes from suppressed sections:
def add_quotes(files):

    logging.info('Using regex to extract quotations from suppressed sections:')

    # Helper method to improve quote matching:
    def clean_quote(text):
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

    def extract_quotes(row):

        # Check if the row relates to a query. If not, can return empty list (as the suppressions are not used in target cases):
        if not (row['query']):
            return []

        quotations = set()
        for ss in row['suppressed_sections']:
            # Find all quotations in the current string
            rule = r'"(.*?)"|that:\s*(.*?)$|said:\s*(.*?)$|observed:\s*(.*?)$|stated:\s*(.*?)$|provides:\s*(.*?)$|noted:\s*(.*?)$|reported:\s*(.*?)$|declared:\s*(.*?)$|explained:\s*(.*?)$|acknowledged:\s*(.*?)$|articulated:\s*(.*?)$|affirmed:\s*(.*?)$|pronounced:\s*(.*?)$|recounted:\s*(.*?)$|described:\s*(.*?)$|elucidated:\s*(.*?)$|clarified:\s*(.*?)$|illustrated:\s*(.*?)$'
            found_quotations = re.findall(rule, ss, re.MULTILINE)
            for quote in found_quotations:
                # Use the first non-None value, or a blank string if all are None
                quote = next(filter(None, quote), '')
                # Check if the quotation has more than 5 words
                if quote and len(quote.split()) > 5:
                    quotations.add(clean_quote(quote))
        return list(quotations)

    tqdm.pandas(desc="Extracting quotes from suppressed sections:")
    files['quotes'] = files.progress_apply(extract_quotes, axis=1)

    logging.info('Added quotes to the "quotes" field for query cases.')



# Method to get entities from English sentences:
def add_entities(files):

    logging.info('Using spacy to extract noun entities, from English sentences:')

    # Helper function to tokenize, lower and remove stop words (for better matching of entities):
    # join used zero space '' to ensure entity is considered as a whole during tfidf
    def process_tokens(text):
        words = text.split()
        return ''.join([word.lower() for word in words if word.isalpha() and word.lower() not in ENGLISH_STOP_WORDS])

    def get_entities(sentences_en):
        text = ' '.join(sentences_en)
        entities = []
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["EVENT", "GPE", "LAW", "LOC", "NORP", "ORG"]:
                processed_tokens = process_tokens(ent.text)
                entities.append(processed_tokens)
        entity_string = ' '.join(entities)
        entity_set = set(entities)

        return entity_string, entity_set

    tqdm.pandas(desc="Extracting entities from english sentences")
    files[['entity_string','entity_set']] = files['sentences_en'].progress_apply(lambda x: get_entities(x)).apply(pd.Series)

    logging.info('Added entity strings to "entity_string" and entities as sets to "entity_set".')



# Method to add english word strings (for tfidf) and sets (for jaccard)
def add_strings_sets(files):

    logging.info('Extracting case word strings (for tfidf) and sets (for case jaccard):')

    def process_string(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return ' '.join([stemmer.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in ENGLISH_STOP_WORDS])

    def get_strings_sets(sections):

        if len(sections) == 0:
            return '', set()

        en_word_string = ' '.join([process_string(s) for s in sections])
        en_word_set = set(en_word_string.split())

        return en_word_string, en_word_set

    tqdm.pandas(desc="Extracting case string and sets from sentences_en:")
    files[['sentences_en_string','sentences_en_set']] = files['sentences_en'].progress_apply(lambda x: get_strings_sets(x)).apply(pd.Series)

    logging.info('Added case strings and sets.')


# Method to add lists of sets
def add_set_lists(files):

    logging.info('Extracting set lists:')

    def process_string(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        words = text.split()
        return set([stemmer.stem(word.lower()) for word in words if word.isalpha() and word.lower() not in ENGLISH_STOP_WORDS])

    def get_set_list(sections):

        set_list = []
        for s in sections:
            set_list.append(process_string(s))

        return set_list

    tqdm.pandas(desc="Extracting set list sentences_en:")
    files['sentences_en_set_list'] = files['sentences_en'].progress_apply(get_set_list)

    tqdm.pandas(desc="Extracting set list paragraphs_formatted:")
    files['paragraphs_formatted_en_set_list'] = files['paragraphs_formatted'].progress_apply(get_set_list)

    tqdm.pandas(desc="Extracting set list from propositions_en:")
    files['propositions_en_set_list'] = files['propositions_en'].progress_apply(get_set_list)

    logging.info('Added set lists.')



# Method to extract judge surname from first paragraph
def add_judge_name(files):

    logging.info('Using regex to extract judge surname from first paragraphs:')

    def extract_judge_surname(paragraphs):

        if len(paragraphs) == 0:
            return None
        else:
            text = paragraphs[0]

        # Regex pattern to match the required format
        pattern = r"\[1\] ([\w'\- ]+),"

        # Find the first match using the regex pattern
        match = re.search(pattern, text)

        # Extract and return the surname if a match is found, otherwise return None
        return match.group(1) if match else None

    tqdm.pandas(desc="Extracting judge surname from paragraphs")
    files['judge'] = files['paragraphs'].progress_apply(extract_judge_surname)

    logging.info('Added judge name to "judge" field.')


# Method to extract year from file:
def add_year(files):

    logging.info('Using string search to find year:')

    # Get most recent year
    def find_most_recent_year(text):
        # Regular expression to match years between 1900 and 2023
        pattern = r'\b(19[0-9]{2}|20[0-1][0-9]|202[0-3])\b'

        # Find all matches
        years = re.findall(pattern, text)

        # Convert to integers and find the max (most recent) year
        if years:
            years = [int(year) for year in years]
            return max(years)
        else:
            return 2024

    tqdm.pandas(desc="Extracting most recent year from file text")
    files['year'] = files['text'].progress_apply(find_most_recent_year)

    logging.info('Added year to files.')



# Method to generate embeddings for sent para combos:
def get_embeddings(files):

    logging.info('Getting embeddings from sentences, paragraphs and propositions.')

    # Helper class for custom dataset, to generate embeddings:
    class PropositionDataset(Dataset):
        def __init__(self, propositions):
            self.propositions = propositions

        def __len__(self):
            return len(self.propositions)

        def __getitem__(self, idx):
            return self.propositions[idx]

    # Function to generate embeddings in batches with progress bar
    def get_embedding_list(dataset, tokenizer, model, batch_size=32):

        dataloader = DataLoader(dataset, batch_size=batch_size)
        embeddings = torch.Tensor().to(device)

        for batch in dataloader:
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            embeddings = torch.cat((embeddings, outputs.last_hidden_state[:, 0, :]), dim=0)

        embeddings_np = embeddings.cpu().numpy()
        embeddings_list = [embedding for embedding in embeddings_np]

        return embeddings_list

    model_name = 'sentence-transformers/all-mpnet-base-v2'

    # Initialize tokenizer and model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)

    def get_embeddings_sentences_en(row):
        return get_embedding_list(PropositionDataset(row['sentences_en']), tokenizer, model)
    tqdm.pandas(desc="Getting embeddings for sentences_en")
    files['embeddings_sentences_en'] = files.progress_apply(get_embeddings_sentences_en, axis=1)

    def get_embeddings_paragraphs_formatted(row):
        return get_embedding_list(PropositionDataset(row['paragraphs_formatted']), tokenizer, model)
    tqdm.pandas(desc="Getting embeddings for paragraphs formatted")
    files['embeddings_paragraphs_formatted'] = files.progress_apply(get_embeddings_paragraphs_formatted, axis=1)

    def get_embeddings_propositions_en(row):
        return get_embedding_list(PropositionDataset(row['propositions_en']), tokenizer, model)
    tqdm.pandas(desc="Getting embeddings for propositions_en")
    files['embeddings_propositions_en'] = files.progress_apply(get_embeddings_propositions_en, axis=1)

    logging.info('Added embeddings.')

