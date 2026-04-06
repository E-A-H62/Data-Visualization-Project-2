import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import RegexpTokenizer, pos_tag
from wordcloud import WordCloud


# Read csv into dataset
df = pd.read_csv("data/children_stories_dataset.csv")

# PREPROCESS STORIES ----------------------------------------------------------
print("\nPreprocessing stories...")

# REMOVE STOP WORDS
stopwords = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'\w+')
df["processed_stopwords"] = df["text"].apply(
    lambda x: " ".join(word.lower() for word in tokenizer.tokenize(x) if word.lower() not in stopwords)
)
print(df["processed_stopwords"].head())
print(df["processed_stopwords"].tail())

# LEMMATIZATION
lemmatizer = WordNetLemmatizer()
tokens = word_tokenize(df["processed_stopwords"].str.cat(sep=" ")) # Tokenize all processed stories into single list of words

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_map = {"J": wordnet.ADJ, "V": wordnet.VERB, "N": wordnet.NOUN, "R": wordnet.ADV}
    return tag_map.get(tag, wordnet.NOUN)

df["lemmatized_text"] = df["processed_stopwords"].apply(
    lambda x: " ".join(lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in word_tokenize(x))
)
print(df["lemmatized_text"].head())
print(df['lemmatized_text'].tail())


# DOUBLE CHECK PREPROCESSING
# Check that processed stories have fewer words than unprocessed stories
# Unprocessed
freq_raw = FreqDist(word_tokenize(df["text"].str.cat(sep=" ")))
print("Unprocessed:", freq_raw.most_common(20))

# Stopwords removed
freq_processed = FreqDist(word_tokenize(df["processed_stopwords"].str.cat(sep=" ")))
print("Stopwords removed:", freq_processed.most_common(20))

# Lemmatized
freq_lemmatized = FreqDist(word_tokenize(df["lemmatized_text"].str.cat(sep=" ")))
print("Lemmatized:", freq_lemmatized.most_common(20))


# VISUALIZE STORIES --------------------------------------------------------------
# Unprocessed wordcloud:
# Expected: Wordclouds of non-preprocessed stories will have more words and more unique words (e.g. "The", "and", "of").
print("\nCreating wordcloud of all stories (unprocessed)...")
unprocessed_wordcloud = WordCloud(background_color="white", stopwords=None, random_state=42).generate(df["text"].str.cat(sep=" ")) # .str.cat(sep=" ") concatenates stories into single string
plt.title("Wordcloud of all stories (unprocessed)")
plt.imshow(unprocessed_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Processed wordcloud:
# Expected: Wordclouds of preprocessed stories will have fewer words and more common words (e.g. "happy", "prince").
print("\nCreating wordcloud of all stories (preprocessed stop words)...")
processed_wordcloud = WordCloud(background_color="white", stopwords=None, random_state=42).generate(df["processed_stopwords"].str.cat(sep=" "))
plt.title("Wordcloud of all stories (preprocessed stop words)")
plt.imshow(processed_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# Expected: Wordclouds of lemmatized stories will have fewer words and more common words (e.g. "happy", "prince").
print("\nCreating wordcloud of all stories (lemmatized)...")
lemmatized_wordcloud = WordCloud(background_color="white", stopwords=None, random_state=42).generate(df["lemmatized_text"].str.cat(sep=" "))
plt.title("Wordcloud of all stories (lemmatized)")
plt.imshow(lemmatized_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()


# Wordclouds of "The Happy Prince" story:
print("\nCreating wordcloud of 'The Happy Prince' story (unprocessed)...")
unprocessed_wcStory1 = WordCloud(background_color="white", stopwords=None, random_state=42).generate(df[df["title"] == "The Happy Prince"]["text"].str.cat(sep=" "))
plt.title("Wordcloud of 'The Happy Prince' story (unprocessed)")
plt.imshow(unprocessed_wcStory1, interpolation="bilinear")
plt.axis("off")
plt.show()

print("\nCreating wordcloud of 'The Happy Prince' story (removed stop words)...")
processed_wcStory1 = WordCloud(background_color="white", stopwords=None, random_state=42).generate(df[df["title"] == "The Happy Prince"]["processed_stopwords"].str.cat(sep=" "))
plt.title("Wordcloud of 'The Happy Prince' story (removed stop words)")
plt.imshow(processed_wcStory1, interpolation="bilinear")
plt.axis("off")
plt.show()

print("\nCreating wordcloud of 'The Happy Prince' story (lemmatized)...")
lemmatized_wcStory1 = WordCloud(background_color="white", stopwords=None, random_state=42).generate(df[df["title"] == "The Happy Prince"]["lemmatized_text"].str.cat(sep=" "))
plt.title("Wordcloud of 'The Happy Prince' story (lemmatized)")
plt.imshow(lemmatized_wcStory1, interpolation="bilinear")
plt.axis("off")
plt.show()