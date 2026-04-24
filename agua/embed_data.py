from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm.auto import tqdm
import re
from sklearn.metrics.pairwise import cosine_similarity

_model_cache: dict = {}


def _load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def get_embedding(text: str, model: str) -> np.ndarray:
    """Return embedding vector for given ``text`` using a sentence-transformers model.

    Parameters
    ----------
    text : str
        Input text to embed.
    model : str
        Sentence-transformers model identifier (e.g. ``"BAAI/bge-m3"``).

    Returns
    -------
    np.ndarray
        1D embedding vector (float64).
    """
    pattern = r"[\t*-]"
    text = re.sub(pattern=pattern, repl="", string=text)
    text = text.replace("\n", " ")
    st_model = _load_model(model)
    emb = st_model.encode(text, normalize_embeddings=True)
    return np.array(emb, dtype=np.float64, copy=True)


def save_sample_embeddings(desc_path, embed_path, doc_embedding_model: str) -> None:
    """Generate embeddings for each description file under ``desc_path``.

    Skips any samples that already have a corresponding ``.npz`` in
    ``embed_path``.

    Parameters
    ----------
    desc_path : PathLike
        Directory containing ``*.txt`` description files.
    embed_path : PathLike
        Output directory for ``.npz`` embedding files.
    doc_embedding_model : str
        Sentence-transformers model name used for description texts.
    """
    files = list(desc_path.glob("*.txt"))
    for file in tqdm(files, desc="Querying embedding for sample", leave=True):
        with open(file, "r") as f:
            description = "\n".join(f.readlines())
            separated_description = description.strip().split("\n")
            filtered_description = "\n".join(separated_description[:-1])
        embedding_file = embed_path / f"{file.stem}.npz"
        embedding_file = embedding_file.resolve()
        embedding_file.parent.mkdir(parents=True, exist_ok=True)
        if not embedding_file.exists():
            embedding = get_embedding(text=filtered_description, model=doc_embedding_model)
            np.savez(embedding_file, embedding=embedding)


def save_concept_embeddings(concepts_file, concept_embedding_save_path, query_embedding_model: str) -> None:
    """Generate embeddings for each concept listed in ``concepts_file``.

    Parameters
    ----------
    concepts_file : PathLike
        Text file with one concept per line, formatted as ``<id>.<text>``.
    concept_embedding_save_path : PathLike
        Output directory for concept embedding ``.npz`` files.
    query_embedding_model : str
        Sentence-transformers model name used for concepts.
    """
    concepts = []
    with open(concepts_file, "r") as f:
        for line in f:
            concept_id, *rest = line.strip().split(".")
            try:
                concept_id = int(concept_id)
            except ValueError as exc:
                raise ValueError("Concept id is not valid. Is the format of the concepts correct?") from exc
            concept = ".".join(rest)
            concepts.append([concept_id, concept])
    for concept_id, concept in tqdm(concepts, leave=True, desc="Querying embedding for concept"):
        embedding_file = concept_embedding_save_path / f"concept_{concept_id:04d}.npz"
        embedding_file.parent.mkdir(parents=True, exist_ok=True)
        if not embedding_file.exists():
            embedding = get_embedding(text=concept, model=query_embedding_model)
            np.savez(embedding_file, embedding=embedding)


def filter_concepts(concept_embedding_save_path, max_intra_concept_similarity: float) -> None:
    """Remove concept embeddings that are too similar to earlier ones.

    Concepts are compared using pairwise cosine similarity; any concept with
    similarity >= ``max_intra_concept_similarity`` to a previous concept is
    deleted (earlier concepts are kept).

    Parameters
    ----------
    concept_embedding_save_path : PathLike
        Directory containing concept embedding ``.npz`` files.
    max_intra_concept_similarity : float
        Threshold above which a concept is considered redundant.
    """
    all_concept_files = sorted(list(concept_embedding_save_path.iterdir()))
    concept_matrix = []
    for concept_file in all_concept_files:
        data = np.load(concept_file)
        embed = data["embedding"]
        concept_matrix.append(embed)
    concept_matrix = np.array(concept_matrix)
    sim_scores = cosine_similarity(concept_matrix)
    concepts_to_remove = []
    for concept_id in range(len(all_concept_files)):
        to_remove = False
        for prev_concept_id in range(concept_id - 1, -1, -1):
            if sim_scores[concept_id, prev_concept_id] >= max_intra_concept_similarity:
                to_remove = True
        if to_remove:
            concepts_to_remove.append(all_concept_files[concept_id])
    for concept_file in concepts_to_remove:
        concept_file.unlink()

