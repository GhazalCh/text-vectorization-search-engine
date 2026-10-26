# Text Vectorization and Search Engine

A Python project demonstrating text vectorization techniques and a simple document retrieval system using TF-IDF and cosine similarity.

## Overview

This project explores how text documents can be represented as numerical vectors and searched using a text query. It demonstrates Binary vectorization, Term Frequency (TF), a custom Inverse Document Frequency (IDF) calculation, and TF-IDF. The search component ranks documents by cosine similarity between the query vector and each document vector.

## Features

- **Binary Vectorization:** Represents whether each term appears in a document.
- **Term Frequency (TF):** Counts term occurrences in each document.
- **Custom IDF Calculation:** Calculates an IDF score using the formula retained from the original coursework implementation.
- **TF-IDF Vectorization:** Uses scikit-learn's `TfidfVectorizer` to create weighted document vectors.
- **Query-Based Search:** Accepts a search query from the command line.
- **Cosine Similarity:** Compares the query vector with document vectors.
- **Ranked Results:** Displays up to five documents in descending similarity order.
- **User-Provided Documents:** Reads `.txt` files from the local `data/` directory; sample documents are not required in the repository.

## Project Structure

```text
text-vectorization-search-engine/
├── data/
│   └── .gitkeep
├── src/
│   └── search_engine.py
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Requirements

- Python 3.8 or later
- NumPy
- pandas
- scikit-learn

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/GhazalCh/text-vectorization-search-engine.git
cd text-vectorization-search-engine
```

Replace `GhazalCh` with your GitHub username.

### 2. (Recommended) Create a virtual environment

**Windows:**

```bash
py -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Add Your Own Documents

This repository does not need to include a sample corpus. You can use your own text documents instead.

1. Open the project's `data/` directory.
2. Add one or more text files with the `.txt` extension.
3. Save the files using UTF-8 encoding.
4. Run the program as described below.

**You do not need to change the filenames in the code.** The program automatically loads all non-empty `.txt` files from `data/`. You can add, remove, or rename documents in that directory, then run the program again.

Example local layout:

```text
data/
├── article_one.txt
├── notes.txt
└── research_topic.txt
```

The filenames above are examples only; use any filenames you prefer. Keep private or copyrighted documents out of a public repository unless you have permission to publish them. The `data/` directory is excluded from Git by default, so your personal documents will not be uploaded accidentally.

## Run the Search Engine

From the project root, run:

```bash
python src/search_engine.py
```

The program prints the Binary, TF, and TF-IDF matrices, displays the top terms according to the custom IDF calculation, and then prompts you to enter a query. It returns up to five documents ranked by cosine similarity.

## How It Works

1. **Load documents:** Read non-empty `.txt` files from `data/`.
2. **Binary vectorization:** Represent each term as present (`1`) or absent (`0`) in each document.
3. **TF vectorization:** Count the occurrences of each term.
4. **Custom IDF calculation:** Calculate term scores using the formula below.
5. **TF-IDF vectorization:** Use scikit-learn to produce TF-IDF document vectors.
6. **Query vectorization:** Transform the user's query using the fitted TF-IDF vectorizer.
7. **Similarity ranking:** Calculate cosine similarity and show the most similar documents.

### Custom IDF formula

The custom IDF calculation preserved from the original implementation is:

`IDF(term) = log(N / (df(term) + 1))`

where:
- `N` is the number of documents.
- `df(term)` is the number of documents containing the term.

This custom formula is separate from scikit-learn's `TfidfVectorizer` calculation, so the custom IDF scores and the TF-IDF matrix may differ.

## Limitations

- Search is based on term overlap, not semantic meaning.
- Text preprocessing follows the default tokenization behavior of scikit-learn's vectorizers.
- The custom IDF calculation splits text on whitespace, so its token boundaries can differ from those used by scikit-learn.
- The program prints the full vectorization matrices; output can become large for large corpora.
- If the query contains only terms not present in the documents, similarity scores will be zero.

## Possible Future Improvements

- Add options for stop-word removal and text normalization.
- Add unit tests for document loading, vectorization, and ranking.
- Make matrix printing optional for large datasets.
- Add precision/recall evaluation using labeled queries and documents.

## License

This project is distributed under the terms specified in the `LICENSE` file.
