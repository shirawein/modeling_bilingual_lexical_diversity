import tensorflow_hub as hub
import tensorflow as tf
!pip install tensorflow-text
import tensorflow_text as text  # Needed for loading universal-sentence-encoder-cmlm/multilingual-preprocess
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import csv

import os

def normalization(embeds):
	norms = np.linalg.norm(embeds, 2, axis=1, keepdims=True)
	return embeds/norms

def total_score(english_lemma_no_dups, arabic_lemma_no_dups):
	score = len(english_lemma_no_dups) + len(arabic_lemma_no_dups)
	return score

def conceptual_score(english_lemma_no_dups, arabic_lemma_no_dups, pairs_above_threshold):
	score = len(english_lemma_no_dups) + len(arabic_lemma_no_dups) - len(pairs_above_threshold)
	return score

def bilingual_score(pairs_above_threshold):
	score = len(pairs_above_threshold)
	return score

def output_top_3_english(english_words_no_dups, arabic_words_no_dups, output):
	for en_word in english_words_no_dups:
 		score = []
 		word = []
 		for ar_word in arabic_words_no_dups:
 			english_sentences = tf.constant([en_word])
 			arabic_sentences = tf.constant([ar_word])

 			english_embeds = encoder(preprocessor(english_sentences))["default"]
 			arabic_embeds = encoder(preprocessor(arabic_sentences))["default"]

 			# For semantic similarity tasks, apply l2 normalization to embeddings
 			english_embeds = normalization(english_embeds)
 			arabic_embeds = normalization(arabic_embeds)

 			sim = np.matmul(english_embeds, np.transpose(arabic_embeds))
 			score.append(sim[0])

 			word.append(ar_word)

 		top3 = sorted(zip(score, word), reverse=True)[:3]
 		print(top3)
 		for sim, arabic_word in top3:
 			output.write(en_word)
 			output.write('\t')
 			output.write(arabic_word)
 			output.write('\t')
 			output.write(str(sim[0]))
 			output.write('\n')

def output_top_3_arabic(english_words_no_dups, arabic_words_no_dups, output):
	for ar_word in arabic_words_no_dups:
		score = []
		word = []
		for en_word in english_words_no_dups:
			english_sentences = tf.constant([en_word])
			arabic_sentences = tf.constant([ar_word])

			english_embeds = encoder(preprocessor(english_sentences))["default"]
			arabic_embeds = encoder(preprocessor(arabic_sentences))["default"]

			# For semantic similarity tasks, apply l2 normalization to embeddings
			english_embeds = normalization(english_embeds)
			arabic_embeds = normalization(arabic_embeds)

			sim = np.matmul(english_embeds, np.transpose(arabic_embeds))
			score.append(sim[0])

			word.append(en_word)

		top3 = sorted(zip(score, word), reverse=True)[:3]
		print(top3)
		for sim, english_word in top3:
			output.write(english_word)
			output.write('\t')
			output.write(ar_word)
			output.write('\t')
			output.write(str(sim[0]))
			output.write('\n')

def output_all_pairs(english_words_no_dups, arabic_words_no_dups, output):
	for ar_word in arabic_words_no_dups:
		for en_word in english_words_no_dups:
			english_sentences = tf.constant([en_word])
			arabic_sentences = tf.constant([ar_word])

			english_embeds = encoder(preprocessor(english_sentences))["default"]
			arabic_embeds = encoder(preprocessor(arabic_sentences))["default"]

			# For semantic similarity tasks, apply l2 normalization to embeddings
			english_embeds = normalization(english_embeds)
			arabic_embeds = normalization(arabic_embeds)

			word_pair = (en_word,ar_word)
			# print(word_pair)

			sim = np.matmul(english_embeds, np.transpose(arabic_embeds))
			# print(sim)

			output.write(en_word)
			# print(en_word)
			output.write('\t')
			output.write(ar_word)
			# print(ar_word)
			output.write('\t')
			output.write(str(sim))
			# print(sim)
			output.write('\n')

			pairs_and_similarity[word_pair] = sim
			# print(pairs_and_similarity)

def retrieve_all_pairs_above_threshold(english_words_no_dups, arabic_words_no_dups, threshold):
	pairs = []
	current_en_words = []
	current_ar_words = []
	english_duplicates = []
	arabic_duplicates = []

	for ar_word in arabic_words_no_dups:
		for en_word in english_words_no_dups:
			english_sentences = tf.constant([en_word])
			arabic_sentences = tf.constant([ar_word])

			english_embeds = encoder(preprocessor(english_sentences))["default"]
			arabic_embeds = encoder(preprocessor(arabic_sentences))["default"]

			# For semantic similarity tasks, apply l2 normalization to embeddings
			english_embeds = normalization(english_embeds)
			arabic_embeds = normalization(arabic_embeds)

			word_pair = (en_word,ar_word)

			sim = np.matmul(english_embeds, np.transpose(arabic_embeds))
			# print(sim)

			if(sim > threshold):
				print(word_pair)
				pairs.append(word_pair)
				if (ar_word in current_ar_words):
					arabic_duplicates.append(ar_word)
				else:
					current_ar_words.append(ar_word)

				if (en_word in current_en_words):
					english_duplicates.append(en_word)
				else:
					current_en_words.append(en_word)

	final_english_duplicates = []
	final_arabic_duplicates = []
	for item in arabic_duplicates:
		for eng, ar in pairs:
			if ar == item:
				curr_pair = (eng,ar)
				final_arabic_duplicates.append(curr_pair)

	for item in english_duplicates:
		for eng, ar in pairs:
			if eng == item:
				curr_pair = (eng,ar)
				final_english_duplicates.append(curr_pair)

	print("List of Arabic duplicates: ", final_arabic_duplicates)
	print("List of English duplicates: ", final_english_duplicates)

	return pairs

preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2")
# preprocessor = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/multilingual-preprocess/3")
# preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/google/LaBSE/2")

# replace "english_file" with the file you are using in its location
english = "english_file"

# replace "arabic_file" with the file you are using in its location
arabic = "arabic_file"

# all_output = open("scores.tsv", "w")
# all_output.write("english_csv")
# all_output.write('\t')
# all_output.write("arabic_csv")
# all_output.write('\t')
# all_output.write("total score")
# all_output.write('\t')
# all_output.write("conceptual score")
# all_output.write('\t')
# all_output.write("bilingual score")
# all_output.write('\n')

# for parent, dirnames, filenames in os.walk('EN_files'):
# 	for fn in filenames:
# 		filepath = os.path.join(parent, fn)
# 		english_csv = filepath
# 		arabic_csv = "AR_files/AR-130" + filepath[15:]
# 		print(arabic_csv)
# 		print(english_csv)

# 		all_output.write(english_csv)
# 		all_output.write('\t')
# 		all_output.write(arabic_csv)
# 		all_output.write('\t')	


arabic_tokens = []
english_tokens = []

# replace "arabic_file.csv" with the name of the file you are using
with open("arabic_file.csv", newline='') as acsv:
	arabic_reader = csv.reader(acsv)
	for row in arabic_reader:
		if 'noun' in row[8].lower() or 'adj' in row[8].lower() or 'verb' in row[8].lower():
			arabic_tokens.append(row[9])

acsv.close()

# replace "english_file.csv" with the name of the file you are using
with open("english_file.csv", newline='') as ecsv:
	english_reader = csv.reader(ecsv)
	for row in english_reader:
		if 'noun' in row[8].lower() or 'adj' in row[8].lower() or 'verb' in row[8].lower():
			english_tokens.append(row[9])


ecsv.close()

english_words = [word.lower() for word in english_tokens if word.isalpha()]
english_words_no_dups = []
[english_words_no_dups.append(x) for x in english_words if x not in english_words_no_dups]

arabic_words = [word.lower() for word in arabic_tokens if word.isalpha()]
arabic_words_no_dups = []
[arabic_words_no_dups.append(x) for x in arabic_words if x not in arabic_words_no_dups]


pairs_and_similarity = {}

# output_name = str(filepath[16:]) + "_results.tsv"

# output = open("test_output", "w")

# output_all_pairs(english_words_no_dups,arabic_words_no_dups,output)
# output_top_3_english(english_words_no_dups,arabic_words_no_dups,output)
# output_top_3_arabic(english_words_no_dups,arabic_words_no_dups,output)

print(english_words_no_dups)
print(arabic_words_no_dups)
		
pairs = retrieve_all_pairs_above_threshold(english_words_no_dups,arabic_words_no_dups,0.69)

# print("ENGLISH WORDS NO DUPS")
# print(english_words_no_dups)
# print("ARABIC WORDS NO DUPS")
# print(arabic_words_no_dups)

print("Number of unique English lemmas: ", len(english_words_no_dups))
print("Total number of English lemmas: ", len(english_words))
print("Number of unique Arabic lemmas: ", len(arabic_words_no_dups))
print("Total number of Arabic lemmas: ", len(arabic_words))

print("All pairs above threshold: ", pairs)
		
total = total_score(english_words_no_dups,arabic_words_no_dups)
conceptual = conceptual_score(english_words_no_dups,arabic_words_no_dups,pairs)
bilingual = bilingual_score(pairs)

print("Total score: ", str(total))
print("Conceptual score: ", str(conceptual))
print("Bilingual score: ", str(bilingual))

# all_output.write(str(total))
# all_output.write('\t')
# all_output.write(str(conceptual))
# all_output.write('\t')
# all_output.write(str(bilingual))
# all_output.write('\n')

				
# output.close()

# print(sorted(pairs_and_similarity.items(), key=lambda x: (x[1],x[1]), reverse=True))
# all_output.close()