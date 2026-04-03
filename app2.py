import io
import math
import os
import re
import tempfile
from collections import Counter
from functools import lru_cache
import kagglehub
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from shiny import App, Inputs, Outputs, Session, render, ui
from shiny.types import ImgData
from wordcloud import WordCloud

matplotlib.use("Agg")
nltk.download("stopwords", quiet=True)

# ── Constants ─────────────────────────────────────────────────────────────────

KNOWN_TITLES: list[str] = [
    "The Happy Prince", "Andersens Fairy Tales", "The Blue Fairy Book",
    "The Adventures of Pinocchio", "MERRY STORIES AND FUNNY PICTURES",
    "The Wonderful Wizard of Oz", "Our Children", "The Little Lame Prince",
    "The Prince and Betty", "The Adventures of Sherlock Holmes", "Peter Pan",
    "The Jungle Book", "The Adventures of Tom Sawyer", "A Little Princess",
    "Little Women", "Just So Stories", "Moby Dick", "Treasure Island",
    "The Idiot", "A Tale of Two Cities", "My Man Jeeves", "Sense and Sensibility",
    "The Time Machine", "The Velveteen Rabbit", "The Book of Dragons",
    "The Snow Image", "The Magical Mimics in Oz", "A Christmas Hamper",
    "My Fathers Dragon", "The Peace Egg and Other tales", "Indian Why Stories",
    "Folk-Tales of the Khasis", "The Paradise of Children", "Wonder Stories",
    "The Best American Humorous Short Stories", "Hindu Tales from the Sanskrit",
    "The Tale of Johnny Town-Mouse", "The Little Red Hen",
    "East of the Sun and West of the Moon", "Among the Forest People",
    "True Stories of Wonderful Deeds", "English Fairy Tales",
    "Simla Village Tales Or Folk Tales from the Himalayas",
    "Japanese Fairy Tales", "Plain Tales of the North", "The Wind in the Willows",
    "A Wonder Book for Girls and Boys", "Tanglewood Tales", "Goody Two-Shoes",
    "The Marvelous Exploits of Paul Bunyan", "Christmas Every Day and Other Stories",
    "The Childrens Book of Thanksgiving Stories",
]

ALL_STORIES_LABEL = "All Stories"
STORY_CHOICES      = [ALL_STORIES_LABEL] + KNOWN_TITLES
TFIDF_STORY_CHOICES = [ALL_STORIES_LABEL] + KNOWN_TITLES

_OUT_DPI   = 110   # DPI for all rendered images (adequate for web)
_WC_W, _WC_H = 800, 533  # WordCloud canvas size (px)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())

def _save_fig(fig: plt.Figure, dpi: int = _OUT_DPI) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        path = tmp.name
    finally:
        tmp.close()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path

def _dark_fig(w: float = 11, h: float = 7):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")
    ax.tick_params(colors="#B0BEC5", labelsize=10)
    return fig, ax

def _empty_fig(msg: str = "No text available.") -> str:
    fig, ax = _dark_fig()
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=14, color="#B0BEC5")
    ax.axis("off")
    return _save_fig(fig)


# ── Placeholder ───────────────────────────────────────────────────────────────

def _make_placeholder() -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")
    ax.text(0.5, 0.5, "Click Generate", ha="center", va="center",
            fontsize=18, color="#B0BEC5")
    ax.axis("off")
    return _save_fig(fig, dpi=80)

_PLACEHOLDER_PATH: str = _make_placeholder()

def _placeholder() -> ImgData:
    # Re-use the single pre-rendered file; copy to a fresh tmp so Shiny's
    # delete_file=True doesn't remove the master after the first serve.
    dst = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    try:
        dst_path = dst.name
    finally:
        dst.close()
    import shutil
    shutil.copy2(_PLACEHOLDER_PATH, dst_path)
    return {"src": dst_path, "alt": "Placeholder", "width": "100%"}


# ── Data loading ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_corpus() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    path = kagglehub.dataset_download("edenbd/children-stories-text-corpus")
    dataset_file = os.path.join(path, "cleaned_merged_fairy_tales_without_eos.txt")

    known_map = {_norm(t): t for t in KNOWN_TITLES}
    known_set = set(known_map)
    stop_words = set(stopwords.words("english"))

    rows: list[dict] = []
    current_title: str | None = None

    with open(dataset_file, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            normalized = _norm(line.strip().rstrip("."))
            if normalized in known_set:
                current_title = known_map[normalized]
                continue
            if current_title and line.strip():
                proc = " ".join(w for w in line.lower().split() if w not in stop_words)
                rows.append({"title": current_title, "text": line.strip(), "processed_text": proc})

    df = pd.DataFrame(rows)

    # Pre-build per-story text strings once
    text_by_story: dict[str, str] = {}
    proc_by_story: dict[str, str] = {}
    for title in KNOWN_TITLES:
        mask = df["title"] == title
        text_by_story[title] = " ".join(df.loc[mask, "text"])
        proc_by_story[title] = " ".join(df.loc[mask, "processed_text"])

    # "All Stories" concatenations
    text_by_story[ALL_STORIES_LABEL] = " ".join(df["text"])
    proc_by_story[ALL_STORIES_LABEL] = " ".join(df["processed_text"])

    return df, text_by_story, proc_by_story

def _get_text(story: str, text_mode: str) -> str:
    _, text_by_story, proc_by_story = _load_corpus()
    return proc_by_story[story] if text_mode == "processed" else text_by_story[story]


# ── TF-IDF corpus ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _build_tfidf_data() -> tuple[
    dict[str, Counter],   # per-story token counters  (stopwords removed)
    dict[str, float],     # IDF per word
    np.ndarray,           # L2-normalised TF-IDF matrix  (n_stories × vocab)
    list[str],            # vocab (sorted)
    list[str],            # story order matching matrix rows
]:
    _, _, proc_by_story = _load_corpus()

    # 1. Build per-story counters
    corpus: dict[str, Counter] = {
        title: Counter(_tokenize(proc_by_story[title]))
        for title in KNOWN_TITLES
    }

    N = len(corpus)

    # 2. Document frequency → smooth IDF
    df_counts: Counter = Counter()
    for cntr in corpus.values():
        for word in cntr:
            df_counts[word] += 1
    idf: dict[str, float] = {
        w: math.log((1 + N) / (1 + df)) + 1
        for w, df in df_counts.items()
    }

    # 3. Build dense TF-IDF matrix (stories × vocab) using numpy
    vocab  = sorted(idf.keys())
    titles = list(corpus.keys())
    word_idx = {w: i for i, w in enumerate(vocab)}
    V, D   = len(vocab), len(titles)

    mat = np.zeros((D, V), dtype=np.float32)
    idf_arr = np.array([idf[w] for w in vocab], dtype=np.float32)

    for r, title in enumerate(titles):
        cntr  = corpus[title]
        total = sum(cntr.values()) or 1
        for word, cnt in cntr.items():
            if word in word_idx:
                mat[r, word_idx[word]] = (cnt / total) * idf_arr[word_idx[word]]

    # 4. L2-normalise for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat_norm = mat / norms

    return corpus, idf, mat_norm, vocab, titles


# ── Word Cloud ────────────────────────────────────────────────────────────────

def get_wordcloud_png_path(story: str, text_mode: str) -> str:
    text = _get_text(story, text_mode)
    wc = WordCloud(background_color="white", random_state=42, width=_WC_W, height=_WC_H).generate(text)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return _save_fig(fig)


# ── Zipf's Law ────────────────────────────────────────────────────────────────

def get_zipf_png_path(story: str, text_mode: str, top_n: int = 50) -> str:
    tokens = _tokenize(_get_text(story, text_mode))
    if not tokens:
        return _empty_fig()

    counts = Counter(tokens)
    ranked = counts.most_common()
    ranks  = np.arange(1, len(ranked) + 1, dtype=float)
    freqs  = np.array([f for _, f in ranked], dtype=float)

    fig, ax = _dark_fig()
    ax.loglog(ranks, freqs, "o", color="#4FC3F7", markersize=3, alpha=0.7, 
            label="Observed word frequency", zorder=3)
    ax.loglog(ranks, freqs[0] / ranks, "--", color="#FF7043", linewidth=2,
            alpha=0.9, label="Zipf's Law (ideal: freq \u221d 1/rank)", zorder=2)
    for i in range(min(top_n, len(ranked))):
        ax.annotate(ranked[i][0], xy=(ranks[i], freqs[i]), xytext=(4, 2),
                    textcoords="offset points", fontsize=6.5, color="#B0BEC5", alpha=0.85)

    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Rank (log scale)", color="#B0BEC5", fontsize=12, labelpad=8)
    ax.set_ylabel("Frequency (log scale)", color="#B0BEC5", fontsize=12, labelpad=8)
    mode_label = "stopwords removed" if text_mode == "processed" else "stopwords kept"
    ax.set_title(f"Zipf's Law \u2014 {story}  ({mode_label})",
                color="#ECEFF1", fontsize=14, fontweight="bold", pad=14)
    ax.legend(fontsize=10, facecolor="#1a1d26", edgecolor="#2a2d3a", labelcolor="#B0BEC5")
    ax.annotate("Zipf's Law: the nth most common word appears ~1/n as often as the most common.",
                xy=(0.01, 0.02), xycoords="axes fraction", fontsize=8,
                color="#78909C", style="italic")
    plt.tight_layout()
    return _save_fig(fig)


# ── Term Frequency ────────────────────────────────────────────────────────────

def get_tf_png_path(story: str, text_mode: str, top_n: int = 30) -> str:
    tokens = _tokenize(_get_text(story, text_mode))
    if not tokens:
        return _empty_fig()

    counts = Counter(tokens)
    total  = len(tokens)
    top    = counts.most_common(top_n)
    words      = [w for w, _ in reversed(top)]
    tfs        = [c / total * 100 for _, c in reversed(top)]
    raw_counts = [c for _, c in reversed(top)]

    fig, ax = _dark_fig(w=11, h=max(6, top_n * 0.32))
    colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(words)))
    bars = ax.barh(words, tfs, color=colors, edgecolor="none", height=0.7)
    gap  = max(tfs) * 0.01
    for bar, cnt in zip(bars, raw_counts):
        ax.text(bar.get_width() + gap, bar.get_y() + bar.get_height() / 2,
                f"{cnt:,}", va="center", ha="left", fontsize=7.5, color="#90A4AE")

    ax.set_xlabel("Term Frequency (%)", color="#B0BEC5", fontsize=11, labelpad=8)
    ax.tick_params(axis="y", labelsize=9, colors="#CFD8DC")
    mode_label = "stopwords removed" if text_mode == "processed" else "stopwords kept"
    ax.set_title(f"Top {top_n} Term Frequencies \u2014 {story}  ({mode_label})",
                color="#ECEFF1", fontsize=13, fontweight="bold", pad=12)
    ax.annotate("TF = (word count) / (total words) \u00d7 100",
                xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                fontsize=8, color="#78909C", style="italic")
    plt.tight_layout()
    return _save_fig(fig)


# ── TF-IDF bar chart ──────────────────────────────────────────────────────────

def get_tfidf_png_path(story: str, top_n: int = 30) -> str:
    corpus, idf, mat_norm, vocab, titles = _build_tfidf_data()
    word_idx = {w: i for i, w in enumerate(vocab)}

    if story == ALL_STORIES_LABEL:
        V = len(vocab)
        idf_arr = np.array([idf[w] for w in vocab], dtype=np.float32)
        sum_scores = np.zeros(V, dtype=np.float64)
        for title in titles:
            cntr  = corpus[title]
            total = sum(cntr.values()) or 1
            for word, cnt in cntr.items():
                if word in word_idx:
                    sum_scores[word_idx[word]] += (cnt / total) * idf_arr[word_idx[word]]
        avg = sum_scores / len(titles)
        top_idx = np.argpartition(avg, -top_n)[-top_n:]
        top_idx = top_idx[np.argsort(avg[top_idx])]   # ascending for chart
        words = [vocab[i] for i in top_idx]
        vals  = [float(avg[i]) for i in top_idx]
        annots = [f"avg cnt: {sum(c.get(vocab[i], 0) for c in corpus.values()) / len(titles):.1f}"
                for i in top_idx]
        chart_title = f"Top {top_n} TF-IDF Terms \u2014 All Stories (averaged)"
        subtitle = "Average TF-IDF score per word across all stories."
    else:
        if story not in corpus:
            return _empty_fig("Story not found.")
        cntr  = corpus[story]
        total = sum(cntr.values()) or 1
        scores = {w: (cnt / total) * idf.get(w, 1.0) for w, cnt in cntr.items()}
        top    = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        words  = [w for w, _ in reversed(top)]
        vals   = [v for _, v in reversed(top)]
        annots = [f"count: {cntr[w]:,}" for w in words]
        chart_title = f"Top {top_n} TF-IDF Terms \u2014 {story}"
        subtitle = "TF-IDF highlights words frequent in this story but rare across all others."

    fig, ax = _dark_fig(w=11, h=max(6, top_n * 0.32))
    colors = plt.cm.YlOrRd(np.linspace(0.35, 0.95, len(words)))
    bars = ax.barh(words, vals, color=colors, edgecolor="none", height=0.7)
    gap  = max(vals) * 0.01
    for bar, ann in zip(bars, annots):
        ax.text(bar.get_width() + gap, bar.get_y() + bar.get_height() / 2,
                ann, va="center", ha="left", fontsize=7.5, color="#90A4AE")

    ax.set_xlabel("TF-IDF Score", color="#B0BEC5", fontsize=11, labelpad=8)
    ax.tick_params(axis="y", labelsize=9, colors="#CFD8DC")
    ax.set_title(chart_title, color="#ECEFF1", fontsize=13, fontweight="bold", pad=12)
    ax.annotate(subtitle, xy=(0.99, 0.01), xycoords="axes fraction", ha="right",
                fontsize=8, color="#78909C", style="italic")
    plt.tight_layout()
    return _save_fig(fig)


# ── TF-IDF Cosine Similarity Heatmap ─────────────────────────────────────────

@lru_cache(maxsize=1)
def get_similarity_png_path() -> str:
    """Cached — expensive matrix multiply only runs once per server lifetime."""
    _, _, mat_norm, _, titles = _build_tfidf_data()
    sim = mat_norm @ mat_norm.T          # cosine similarity  (n × n)

    # Greedy nearest-neighbour ordering for visual clustering
    n       = len(titles)
    visited = [False] * n
    order   = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = order[-1]
        row  = sim[last]
        best = max((j for j in range(n) if not visited[j]), key=lambda j: row[j])
        order.append(best)
        visited[best] = True

    sim_ord = sim[np.ix_(order, order)]
    labels  = [titles[i] for i in order]
    short   = [t[:27] + "\u2026" if len(t) > 28 else t for t in labels]

    n_s      = len(labels)
    fig_size = max(14, n_s * 0.38)
    fig, ax  = plt.subplots(figsize=(fig_size, fig_size * 0.88))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    im   = ax.imshow(sim_ord, cmap="magma", vmin=0, vmax=1, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(colors="#B0BEC5", labelsize=9)
    cbar.set_label("Cosine Similarity", color="#B0BEC5", fontsize=10)

    ax.set_xticks(range(n_s))
    ax.set_yticks(range(n_s))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7.5, color="#CFD8DC")
    ax.set_yticklabels(short, fontsize=7.5, color="#CFD8DC")
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2d3a")

    ax.set_title("TF-IDF Cosine Similarity Between Stories",
                color="#ECEFF1", fontsize=14, fontweight="bold", pad=16)
    ax.annotate(
        "Brighter = more similar TF-IDF profiles. "
        "Stories ordered by greedy nearest-neighbour clustering.",
        xy=(0.5, -0.18), xycoords="axes fraction", ha="center",
        fontsize=8, color="#78909C", style="italic",
    )
    plt.tight_layout()
    return _save_fig(fig, dpi=130)


# ── UI ────────────────────────────────────────────────────────────────────────

_shared_controls = [
    ui.input_select("story", "Story", choices=STORY_CHOICES, selected=ALL_STORIES_LABEL),
    ui.input_radio_buttons(
        "text_mode", "Text version",
        {"processed": "Processed (stopwords removed)",
        "no preprocess": "No preprocess (stopwords kept)"},
        selected="processed",
    ),
    ui.input_action_button("render_btn", "Generate"),
]

app_ui = ui.page_fluid(
    ui.h2("Children Stories \u2014 Text Visualizations"),
    ui.navset_tab(

        ui.nav_panel("Word Cloud",
            ui.layout_sidebar(
                ui.sidebar(*_shared_controls),
                ui.output_image("wordcloud", height="700px"),
                ui.output_text("wc_caption"),
            ),
        ),

        ui.nav_panel("Zipf's Law",
            ui.layout_sidebar(
                ui.sidebar(
                    *_shared_controls,
                    ui.input_slider("top_n_labels", "Word labels shown",
                                    min=10, max=100, value=50, step=5),
                ),
                ui.output_image("zipf_plot", height="700px"),
                ui.output_text("zipf_caption"),
            ),
        ),

        ui.nav_panel("Term Frequency",
            ui.layout_sidebar(
                ui.sidebar(
                    *_shared_controls,
                    ui.input_slider("tf_top_n", "Words to show",
                                    min=10, max=60, value=30, step=5),
                ),
                ui.output_image("tf_plot", height="750px"),
                ui.output_text("tf_caption"),
            ),
        ),

        ui.nav_panel("TF-IDF",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select("tfidf_story", "Story",
                                    choices=TFIDF_STORY_CHOICES,
                                    selected=ALL_STORIES_LABEL),
                    ui.input_slider("tfidf_top_n", "Words to show",
                                    min=10, max=60, value=30, step=5),
                    ui.input_action_button("render_btn_tfidf", "Generate"),
                    ui.p(ui.tags.small(
                        '"All Stories" shows the average TF-IDF score across the whole corpus. '
                        "Stopwords are always removed for this view."
                    ), style="color:#78909C;margin-top:8px;"),
                ),
                ui.output_image("tfidf_plot", height="750px"),
                ui.output_text("tfidf_caption"),
            ),
        ),

        ui.nav_panel("TF-IDF Similarity",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_action_button("render_btn_sim", "Generate"),
                    ui.p(ui.tags.small(
                        "Pairwise cosine similarity between every story's TF-IDF vector. "
                        "Brighter = more similar vocabulary. "
                    ), style="color:#78909C;margin-top:8px;"),
                ),
                ui.output_image("sim_plot", height="820px"),
                ui.output_text("sim_caption"),
            ),
        ),
    ),
)


# ── Server ────────────────────────────────────────────────────────────────────

def server(input: Inputs, output: Outputs, session: Session):

    @render.image(delete_file=True)
    def wordcloud() -> ImgData:
        if input.render_btn() == 0:
            return _placeholder()
        return {"src": get_wordcloud_png_path(input.story(), input.text_mode()),
                "alt": f"Word cloud \u2014 {input.story()}", "width": "100%"}

    @render.text
    def wc_caption() -> str:
        if input.render_btn() == 0:
            return "Choose a story, then click 'Generate'."
        mode = "stopwords removed" if input.text_mode() == "processed" else "stopwords kept"
        return f"Word cloud ({mode}) for: {input.story()}"

    @render.image(delete_file=True)
    def zipf_plot() -> ImgData:
        if input.render_btn() == 0:
            return _placeholder()
        return {"src": get_zipf_png_path(input.story(), input.text_mode(),
                                        top_n=input.top_n_labels()),
                "alt": f"Zipf \u2014 {input.story()}", "width": "100%"}

    @render.text
    def zipf_caption() -> str:
        if input.render_btn() == 0:
            return "Choose a story, then click 'Generate'."
        mode = "stopwords removed" if input.text_mode() == "processed" else "stopwords kept"
        return (f"Log-log rank vs. frequency ({mode}) for: {input.story()}. "
                "Orange dashed line = ideal Zipf distribution.")

    @render.image(delete_file=True)
    def tf_plot() -> ImgData:
        if input.render_btn() == 0:
            return _placeholder()
        return {"src": get_tf_png_path(input.story(), input.text_mode(),
                                        top_n=input.tf_top_n()),
                "alt": f"TF \u2014 {input.story()}", "width": "100%"}

    @render.text
    def tf_caption() -> str:
        if input.render_btn() == 0:
            return "Choose a story, then click 'Generate'."
        mode = "stopwords removed" if input.text_mode() == "processed" else "stopwords kept"
        return (f"Top {input.tf_top_n()} words by relative term frequency "
                f"({mode}) for: {input.story()}. Raw counts shown beside each bar.")

    @render.image(delete_file=True)
    def tfidf_plot() -> ImgData:
        if input.render_btn_tfidf() == 0:
            return _placeholder()
        return {"src": get_tfidf_png_path(input.tfidf_story(), top_n=input.tfidf_top_n()),
                "alt": f"TF-IDF \u2014 {input.tfidf_story()}", "width": "100%"}

    @render.text
    def tfidf_caption() -> str:
        if input.render_btn_tfidf() == 0:
            return "Choose a story, then click 'Generate'."
        if input.tfidf_story() == ALL_STORIES_LABEL:
            return (f"Top {input.tfidf_top_n()} words by average TF-IDF across all stories. "
                    "These words are consistently distinctive across the corpus.")
        return (f"Top {input.tfidf_top_n()} TF-IDF words for: {input.tfidf_story()}. "
                "Words scoring highest are frequent here but rare across the other stories.")

    @render.image(delete_file=True)
    def sim_plot() -> ImgData:
        if input.render_btn_sim() == 0:
            return _placeholder()
        return {"src": get_similarity_png_path(),
                "alt": "TF-IDF cosine similarity heatmap", "width": "100%"}

    @render.text
    def sim_caption() -> str:
        if input.render_btn_sim() == 0:
            return "Click 'Generate' to compute pairwise TF-IDF similarity across all stories."
        return ("Pairwise cosine similarity of TF-IDF vectors. "
                "1.0 (brightest) = identical vocabulary profiles; "
                "0.0 (darkest) = no shared distinctive vocabulary.")


app = App(app_ui, server)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
