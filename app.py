import os
import re
import tempfile
from functools import lru_cache
import nltk
from nltk.corpus import stopwords
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud
import kagglehub
from shiny import App, Inputs, Outputs, Session, render, ui
from shiny.types import ImgData
# Shiny runs on a server; avoid requiring a GUI backend.
matplotlib.use("Agg")

"""
Story titles not explicitly listed in text corpus:
Myths Retold by Children
Household Tales
Indian Fairy Tales
Fairy Tales Second Series
Childhoods Favorites and Fairy Stories
Celtic Tales
The Secret Garden
Comic History of the United States
Folk Tales from the Russian
Snow-White or The House in the Wood
Dramatic Reader for Lower Grades
Aesop Fables
The Louisa Alcott Reader…
The Pig Brother and Other Fables and Stories
The Worlds Greatest Books, Volume 3
"""

# Canonical story titles used to detect section headers in the raw text file.
KNOWN_TITLES = [
    "The Happy Prince",
    "Andersens Fairy Tales",
    "The Blue Fairy Book",
    "The Adventures of Pinocchio",
    "MERRY STORIES AND FUNNY PICTURES",
    "The Wonderful Wizard of Oz",
    "Our Children",
    "The Little Lame Prince",
    "The Prince and Betty",
    "The Adventures of Sherlock Holmes",
    "Peter Pan",
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
    "The Velveteen Rabbit",
    "The Book of Dragons",
    "The Snow Image",
    "The Magical Mimics in Oz",
    "A Christmas Hamper",
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
    "A Wonder Book for Girls and Boys",
    "Tanglewood Tales",
    "Goody Two-Shoes",
    "The Marvelous Exploits of Paul Bunyan",
    "Christmas Every Day and Other Stories",
    "The Childrens Book of Thanksgiving Stories",
]
ALL_STORIES_LABEL = "All Stories"
DEFAULT_SELECTED_STORY = ALL_STORIES_LABEL
# Include an "All Stories" rollup option before individual titles.
STORY_CHOICES = [ALL_STORIES_LABEL] + KNOWN_TITLES

def _norm(s: str) -> str:
    """Lowercase and remove non-alphanumerics so title lines match."""
    return "".join(ch.lower() for ch in s if ch.isalnum())

@lru_cache(maxsize=1)  # Cache the processed dataframe across reactive calls.
def get_processed_df() -> pd.DataFrame:
    """
    Load dataset and preprocess with stopwords removed.
    """
    # Downloads (and caches locally) the Kaggle dataset bundle.
    path = kagglehub.dataset_download("edenbd/children-stories-text-corpus")
    dataset_file = os.path.join(path, "cleaned_merged_fairy_tales_without_eos.txt")
    # Prepare known title matching (same approach as visualizations.py)
    known_map = {_norm(t): t for t in KNOWN_TITLES}
    known_set = set(known_map.keys())

    # Read dataset
    with open(dataset_file, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.rstrip("\n") for line in f]

    rows: list[dict[str, str]] = []
    current_title: str | None = None

    # Parse dataset into (title, text) rows
    for line in lines:
        # Title lines sometimes vary in punctuation/case; normalize before matching.
        normalized = _norm(line.strip().rstrip("."))
        if normalized in known_set:
            current_title = known_map[normalized]
            continue
        # Treat non-empty lines after a recognized title as story text.
        if current_title is not None and line.strip() != "":
            rows.append({"title": current_title, "text": line.strip()})

    df = pd.DataFrame(rows)
    stop_words = set(stopwords.words("english")) # Remove stop words
    df["processed_text"] = df["text"].apply(lambda x: " ".join(word.lower() for word in x.split() if word.lower() not in stop_words))
    return df


def get_selected_story_text(story: str) -> str:
    # Default to the preprocessed (stopwords removed) version for wordclouds.
    return get_selected_story_text_by_mode(story, text_mode="processed")


def get_selected_story_text_by_mode(story: str, text_mode: str) -> str:
    """
    Build the text used for a wordcloud.
    text_mode:
      - "processed" -> stopwords removed (uses `processed_text`)
      - "no preprocess" -> stopwords kept (uses `text`)
    """
    df = get_processed_df()
    col = "processed_text" if text_mode == "processed" else "text"

    if story == ALL_STORIES_LABEL:
        return df[col].str.cat(sep=" ")
    return df[df["title"] == story][col].str.cat(sep=" ")


def get_wordcloud_png_path(story: str, text_mode: str) -> str:
    """
    Generate a wordcloud PNG file for the selected story.
    This intentionally does not use a persistent cache: it writes to a
    temporary PNG and Shiny will delete it after rendering.
    """
    text = get_selected_story_text_by_mode(story, text_mode)
    # Generate and render wordcloud
    wc = WordCloud(
        background_color="white",
        random_state=42,
        width=1200,
        height=800,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    # Use a temp file because `render.image` expects a file path (not raw bytes).
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp_path = tmp.name
    finally:
        tmp.close()

    fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(tmp_path)


def make_placeholder_image() -> str:
    """Create a small temporary placeholder image."""

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_facecolor("white")
    ax.text(0.5, 0.5, "Click Generate", ha="center", va="center", fontsize=18)
    ax.axis("off")
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        tmp_path = tmp.name
    finally:
        tmp.close()
    fig.savefig(tmp_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return str(tmp_path)

app_ui = ui.page_fluid(
    ui.h2("Children Stories Word Clouds"),
    ui.input_select("story", "Story", choices=STORY_CHOICES, selected=DEFAULT_SELECTED_STORY),
    ui.input_radio_buttons(
        "text_mode",
        "Text version",
        {
            "processed": "Processed (stopwords removed)",
            "no preprocess": "No preprocess (stopwords kept)",
        },
        selected="processed",
    ),
    ui.input_action_button("render", "Generate word cloud"),
    ui.output_image("wordcloud", height="700px"),
    ui.output_text("caption"),
    ui.output_text("status"),
)

def server(input: Inputs, output: Outputs, session: Session):
    @render.text
    def status() -> str:
        if input.render() == 0:
            return "Ready. Click 'Generate word cloud'."
        return "Generating word cloud..."

    @render.image(delete_file=True)
    def wordcloud() -> ImgData:
        # Only generate when the user clicks "Generate" (avoids long initial load).
        # Fresh temp file each time: `delete_file=True` removes the PNG after serve, so a
        # module-level path would break on the next render (reconnect, refresh, etc.).
        if input.render() == 0:
            return {
                "src": make_placeholder_image(),
                "alt": "Word cloud placeholder",
                "width": "100%",
            }

        story = input.story()
        text_mode = input.text_mode()
        path = get_wordcloud_png_path(story, text_mode)
        # Shiny expects a file path for `render.image`
        return {
            "src": path,
            "alt": f"Word cloud for {story} ({text_mode})",
            "width": "100%",
        }

    @render.text
    def caption() -> str:
        story = input.story()
        if input.render() == 0:
            return "Choose a story, then click 'Generate word cloud'."
        text_mode = input.text_mode()
        if story == ALL_STORIES_LABEL:
            return (
                "Word cloud (with preprocessing on text) for all stories."
                if text_mode == "processed"
                else "Word cloud (with no preprocessing on text) for all stories."
            )
        return (
            f"Word cloud (with preprocessing on text) for: {story}"
            if text_mode == "processed"
            else f"Word cloud (with no preprocessing on text) for: {story}"
        )

app = App(app_ui, server)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    # Bind to all interfaces for common PaaS environments.
    app.run(host="0.0.0.0", port=port)