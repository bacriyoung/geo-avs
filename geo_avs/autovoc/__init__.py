"""Auto-vocabulary proposal and verification utilities."""

from .caption2tag import CaptionTagger, flatten_caption_value
from .remote_sensing_lexicon import CANONICAL_TERMS, normalize_term

try:
    from .vocabulary_scoring import score_terms
    from .vocabulary_verification import verify_vocabulary
except ModuleNotFoundError:  # Allows caption-only tools to run without torch.
    score_terms = None
    verify_vocabulary = None

__all__ = [
    "CANONICAL_TERMS",
    "CaptionTagger",
    "flatten_caption_value",
    "normalize_term",
    "score_terms",
    "verify_vocabulary",
]
