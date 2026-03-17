import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud

# CREATE DATASET ----------------------------------------------------
print("Creating dataset...")
# Download latest version
path = kagglehub.dataset_download("edenbd/children-stories-text-corpus")
print("Path to dataset files:", path)
# Path to dataset file
dataset_file = os.path.join(path, "cleaned_merged_fairy_tales_without_eos.txt")

# Known titles of stories in dataset
KNOWN_TITLES = [
    "The Happy Prince",
    "Andersens Fairy Tales",
    "The Blue Fairy Book",
    "The Adventures of Pinocchio",
    "Myths Retold by Children",
    "Household Tales",
    "Indian Fairy Tales",
    "Fairy Tales Second Series",
    "MERRY STORIES AND FUNNY PICTURES",
    "Childhoods Favorites and Fairy Stories",
    "The Wonderful Wizard of Oz",
    "Celtic Tales",
    "Our Children",
    "The Little Lame Prince",
    "The Prince and Betty",
    "The Adventures of Sherlock Holmes",
    "Peter Pan",
    "The Secret Garden",
    "The Jungle Book",
    "The Adventures of Tom Sawyer",
    "A Little Princess",
    "Little Women",
    "Just So Stories",
    "Moby Dick",
    "Treasure Island",
    "The Idiot",
    "A Tale of Two Cities",
    "My Man Jeeves",
    "Sense and Sensibility",
    "The Time Machine",
    "Comic History of the United States",
    "The Velveteen Rabbit",
    "The Book of Dragons",
    "The Snow Image",
    "The Magical Mimics in Oz",
    "Folk Tales from the Russian",
    "Snow-White or The House in the Wood",
    "Dramatic Reader for Lower Grades",
    "A Christmas Hamper",
    "Aesop Fables",
    "My Fathers Dragon",
    "The Peace Egg and Other tales",
    "Indian Why Stories",
    "Folk-Tales of the Khasis",
    "The Paradise of Children",
    "Wonder Stories",
    "The Best American Humorous Short Stories",
    "Hindu Tales from the Sanskrit",
    "The Tale of Johnny Town-Mouse",
    "The Little Red Hen",
    "East of the Sun and West of the Moon",
    "Among the Forest People",
    "True Stories of Wonderful Deeds",
    "English Fairy Tales",
    "Simla Village Tales Or Folk Tales from the Himalayas",
    "Japanese Fairy Tales",
    "Plain Tales of the North",
    "The Wind in the Willows",
    "The Louisa Alcott Reader. A Supplementary Reader for the Fourth Year of School",
    "A Wonder Book for Girls and Boys",
    "Tanglewood Tales",
    "The Pig Brother and Other Fables and Stories",
    "The Worlds Greatest Books, Vol 3",
    "Goody Two-Shoes",
    "The Marvelous Exploits of Paul Bunyan",
    "Christmas Every Day and Other Stories",
    "The Childrens Book of Thanksgiving Stories",
]

def _norm(s: str) -> str:
    """Lowercase and remove non-alphanumerics so lines like 'ANDERSEN'S FAIRY TALES' match 'Andersens Fairy Tales'."""
    return "".join(ch.lower() for ch in s if ch.isalnum())

known_map = {_norm(t): t for t in KNOWN_TITLES} # Dictionary of known titles and their normalized versions
known_set = set(known_map.keys()) # Set of normalized known titles

# Read dataset
with open(dataset_file, "r", encoding="utf-8", errors="replace") as f:
    lines = [line.rstrip("\n") for line in f]

rows = [] # List of rows
current_title = None # Current title

# Process dataset
for line in lines:
    normalized = _norm(line.strip().rstrip(".")) # Normalize line
    # If this line is a known title, switch the current story title.
    if normalized in known_set:
        current_title = known_map[normalized]
        continue
    # Otherwise, treat it as story text belonging to the latest title we've seen.
    if current_title is not None and line.strip() != "":
        rows.append({"title": current_title, "text": line.strip()}) # Add row to list

# Create dataframe from rows
df = pd.DataFrame(rows)

# Print the dataset
print(df.head())
print(df.tail())


# PREPROCESS STORIES ----------------------------------------------------------
print("\nPreprocessing stories...")
# STILL IN PROGRESS


# VISUALIZE STORIES --------------------------------------------------------------
print("\nCreating wordcloud of all stories (not preprocessed)...")
wordcloud = WordCloud(background_color="white").generate(df["text"].str.cat(sep=" ")) # .str.cat(sep=" ") concatenates stories into single string
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

print("\nCreating wordcloud of all stories (preprocessed)...")
# STILL IN PROGRESS

print("\nCreating wordcloud of 'The Happy Prince' story...")
# STILL IN PROGRESS FOR PREPROCESS
wcStory1 = WordCloud(background_color="white").generate(df[df["title"] == "The Happy Prince"]["text"].str.cat(sep=" ")) # .str.cat(sep=" ") concatenates stories into single string
plt.imshow(wcStory1, interpolation="bilinear")
plt.axis("off")
plt.show()