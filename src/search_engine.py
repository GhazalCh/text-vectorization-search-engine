from collections import Counter
from math import log
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Project root is the parent directory of this script's "src" directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TOP_K = 5


def load_documents(folder_path: Path):
    """Load every .txt document in the data directory, in filename order."""
    if not folder_path.exists():
        raise FileNotFoundError(
            f"Data directory was not found: {folder_path}\n"
            "Create a 'data' directory in the project root and add .txt files."
        )

    files = sorted(folder_path.glob("*.txt"))
    documents = []
    file_names = []

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"Could not read '{file_path.name}' as UTF-8. "
                "Save the file with UTF-8 encoding and try again."
            ) from exc

        if text:
            documents.append(text)
            file_names.append(file_path.name)

    if not documents:
        raise ValueError(
            f"No non-empty .txt files were found in {folder_path}.\n"
            "Add at least one UTF-8 encoded text file to the data directory."
        )

    return documents, file_names


def show_matrix(title: str, matrix, feature_names, file_names):
    """Print a vectorization matrix as a readable pandas DataFrame."""
    dataframe = pd.DataFrame(
        matrix,
        index=file_names,
        columns=feature_names,
    )
    dataframe.index.name = "Document"
    print(f"\n{title}")
    print(dataframe.to_string())


def calculate_custom_idf(documents):
    """
    Calculate the project's custom IDF score:
        IDF(term) = log(number_of_documents / (document_frequency + 1))

    This formula is retained from the original coursework implementation.
    It differs from the IDF formula used internally by scikit-learn.
    """
    tokenized_docs = [document.split() for document in documents]
    document_frequency = Counter()

    for tokens in tokenized_docs:
        for token in set(tokens):
            document_frequency[token] += 1

    total_docs = len(tokenized_docs)
    idf_scores = {
        token: log(total_docs / (frequency + 1))
        for token, frequency in document_frequency.items()
    }
    return idf_scores


def main():
    documents, file_names = load_documents(DATA_DIR)
    print(f"Loaded {len(documents)} document(s) from: {DATA_DIR}")

    # 1. Binary vectorization: 1 if a term appears in a document, otherwise 0.
    binary_vectorizer = CountVectorizer(binary=True)
    binary_matrix = binary_vectorizer.fit_transform(documents).toarray()
    show_matrix(
        "Binary Vectorization",
        binary_matrix,
        binary_vectorizer.get_feature_names_out(),
        file_names,
    )

    # 2. Term Frequency (TF): raw term counts.
    tf_vectorizer = CountVectorizer()
    tf_matrix = tf_vectorizer.fit_transform(documents).toarray()
    show_matrix(
        "Term Frequency (TF)",
        tf_matrix,
        tf_vectorizer.get_feature_names_out(),
        file_names,
    )

    # 3. Custom IDF calculation, preserved from the original implementation.
    idf_scores = calculate_custom_idf(documents)
    top_idf = sorted(idf_scores.items(), key=lambda item: item[1], reverse=True)[:10]

    print("\nTop 10 terms by the custom IDF score:")
    for term, score in top_idf:
        print(f"- {term}: {score:.4f}")

    # 4. TF-IDF vectorization using scikit-learn.
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents).toarray()
    show_matrix(
        "TF-IDF Vectorization",
        tfidf_matrix,
        tfidf_vectorizer.get_feature_names_out(),
        file_names,
    )

    # 5. Search documents using cosine similarity between TF-IDF vectors.
    query = input("\nEnter your search query: ").strip()
    if not query:
        print("The query is empty. No search was performed.")
        return

    query_vector = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, tfidf_matrix)[0]
    ranked_indices = similarities.argsort()[::-1][:TOP_K]

    print(f"\nTop {min(TOP_K, len(documents))} related document(s):")
    for rank, index in enumerate(ranked_indices, start=1):
        print(f"\n{rank}. {file_names[index]} (cosine similarity: {similarities[index]:.4f})")
        print(documents[index])


if __name__ == "__main__":
    main()
