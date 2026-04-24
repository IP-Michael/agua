"""Embed concepts and state descriptions for the anomaly detection domain.

Usage
-----
    python embed_data.py --save_concept_embeddings
    python embed_data.py --save_sample_embeddings
    python embed_data.py --save_concept_embeddings --save_sample_embeddings
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from argparse import ArgumentParser
import global_constants as GC
from agua import embed_data as agua_embed


def main() -> None:
    """CLI entry for concept and sample embedding generation.

    Flags
    -----
    --save_concept_embeddings : Embed each concept from GC.CONCEPTS_FILE.
    --save_sample_embeddings  : Embed each saved input description.
    """
    parser = ArgumentParser(
        description="Generate embeddings for anomaly detection concepts and state descriptions."
    )
    parser.add_argument(
        "--save_concept_embeddings",
        action="store_true",
        default=False,
        help="Generate embeddings for each concept in GC.CONCEPTS_FILE.",
    )
    parser.add_argument(
        "--save_sample_embeddings",
        action="store_true",
        default=False,
        help="Generate embeddings for each saved input description file.",
    )
    args = parser.parse_args()

    if args.save_concept_embeddings:
        GC.CONCEPT_EMBEDDING_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        agua_embed.save_concept_embeddings(
            GC.CONCEPTS_FILE,
            GC.CONCEPT_EMBEDDING_SAVE_PATH,
            GC.QUERY_EMBEDDING_MODEL,
        )

    if args.save_sample_embeddings:
        GC.STATE_EMBEDDING_SAVE_PATH.mkdir(parents=True, exist_ok=True)
        agua_embed.save_sample_embeddings(
            GC.STATE_DESCRIPTION_SAVE_PATH,
            GC.STATE_EMBEDDING_SAVE_PATH,
            GC.DOC_EMBEDDING_MODEL,
        )


if __name__ == "__main__":
    main()
