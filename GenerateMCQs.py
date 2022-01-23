# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 11:44:28 2021

@author: Rishav
"""

import re
import torch
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM

# Load pre-trained model tokenizer (vocabulary)
import time
start = time.time()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
end = time.time()
print ("Time Elapsed to load BERT ",end-start)

# Main function to predict the top 30 choices for the fill in the blank word using BERT. 
# Eg: The Sun is more ____ 4 billion years old.

def get_predicted_words(text):
    text = "[CLS] " + text.replace("____", "[MASK]") + " [SEP]"
    # text= '[CLS] Tom has fully [MASK] from his illness. [SEP]'
    tokenized_text = tokenizer.tokenize(text)
    #print("tokenized sentence: ",tokenized_text,"\n")
    masked_index = tokenized_text.index('[MASK]')
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    # Predict all tokens
    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensors)

    # Get 30 choices for the masked(blank) word 
    k = 30
    predicted_index, predicted_index_values = torch.topk(predictions[0, masked_index], k)
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_index_values.tolist())
    filtered_tokens_to_remove_punctuation = []
    # Remove any predictions that contain punctuation etc as they are not relevant to us.
    for token in predicted_tokens:
        if re.match("^[a-zA-Z0-9_]*$", token):
            filtered_tokens_to_remove_punctuation.append(token)
        
    return filtered_tokens_to_remove_punctuation

# Read an article from a file
file_path = "egypt.txt" #other texts in same directory: "PSLE.txt", "hellenkeller.txt", "Grade7_electricity.txt" , "material.txt", "paperboat.txt"

def read_file(file_path):
    with open(file_path, 'rb') as content_file:
        content = content_file.read()
        return content
    
text = read_file(file_path)
text = text.decode("utf-8")
print(text)

#  We will extract some adpositions. An adposition is a cover term for prepositions and postpositions.
import pke
import string

def get_adpositions_multipartite(text):
    out=[]

    extractor = pke.unsupervised.MultipartiteRank()
    extractor.load_document(input=text)
    #    not contain punctuation marks or stopwords as candidates.
    pos = {'ADP'} #Adpositions
    stoplist = list(string.punctuation)
    stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
    extractor.candidate_selection(pos=pos, stoplist=stoplist)
    # 4. build the Multipartite graph and rank candidates using random walk,
    #    alpha controls the weight adjustment mechanism, see TopicRank for
    #    threshold/method parameters.
    extractor.candidate_weighting(alpha=1.1,
                                  threshold=0.75,
                                  method='average')
    keyphrases = extractor.get_n_best(n=10)

    for key in keyphrases:
        out.append(key[0])

    return out


adpositions = get_adpositions_multipartite(text)
print ("Adpositions from the text: ",adpositions)

# Get all the sentences for a given adpostion word. So each word may have mulitple sentences.
from nltk.tokenize import sent_tokenize
import nltk
nltk.download('punkt')
from flashtext import KeywordProcessor

def tokenize_sentences(text):
    sentences = [sent_tokenize(text)]
    sentences = [y for x in sentences for y in x]
    # Remove any short sentences less than 20 letters.
    sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
    return sentences
sentences = tokenize_sentences(text)

def get_sentences_for_keyword(keywords, sentences):
    keyword_processor = KeywordProcessor()
    keyword_sentences = {}
    for word in keywords:
        keyword_sentences[word] = []
        keyword_processor.add_keyword(word)
    for sentence in sentences:
        keywords_found = keyword_processor.extract_keywords(sentence)
        for key in keywords_found:
            keyword_sentences[key].append(sentence)

    for key in keyword_sentences.keys():
        values = keyword_sentences[key]
        values = sorted(values, key=len, reverse=True)
        keyword_sentences[key] = values
    return keyword_sentences

keyword_sentence_mapping_adpos = get_sentences_for_keyword(adpositions, sentences)

for word in keyword_sentence_mapping_adpos:
    print (word, " : ",keyword_sentence_mapping_adpos[word],"\n")
    
#  For every adposition word we have multiple sentences. For every sentence we blank the adposition word and ask BERT 
#  to predict the top N choices. Then make a note of index of the correct answer in the predicitons. Then we sort the
# sentences by the index and pick the top one.
def get_best_sentence_and_options(word, sentences_array):
    keyword = word
    sentences = sentences_array
    sentences = sorted(sentences, key=len, reverse=False)
    max_no = min(5, len(sentences))
    sentences = sentences[:max_no]
    choices_filtered = []
    ordered_sentences = []
    for sentence in sentences:
        insensitive_line = re.compile(re.escape(keyword), re.IGNORECASE)
        no_of_replacements =  len(re.findall(re.escape(keyword),sentence,re.IGNORECASE))
        #blanked_sentence = sentence.replace(keyword, "____", 1)
        blanked_sentence = insensitive_line.sub("____", sentence)
        blanks = get_predicted_words(blanked_sentence)

        if blanks is not None:
            choices_filtered = blanks
            try:
                word_index = choices_filtered.index(keyword.lower())
                if no_of_replacements<2:
                    ordered_sentences.append((blanked_sentence, choices_filtered, word_index))
            except:
                pass

    ordered_sentences = sorted(ordered_sentences, key=lambda x: x[2])
    if len(ordered_sentences) > 0:
        return (ordered_sentences[0][0], ordered_sentences[0][1])
    else:
        return None, None
    
for each_adpos in adpositions:
    sentence, best_options = get_best_sentence_and_options(each_adpos, keyword_sentence_mapping_adpos[each_adpos])
    print (sentence)
    print (best_options)
    print ("\n\n")