"""
polyBERT Predictor for Polymer Properties (Tier 3)

Uses kuelumbus/polyBERT (DeBERTa-based, ~2GB) from HuggingFace
to generate 600-D polymer embeddings, then applies lightweight
sklearn prediction heads for property prediction.

Reference:
    Kuenneth & Ramprasad, "polyBERT: a chemical language model to enable
    fully machine-driven ultrafast polymer informatics", Nature Communications, 2023.
"""

import logging
import os
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)

# Check if transformers is available
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class PolyBERTPrediction:
    """polyBERT prediction result."""
    success: bool = True
    polymer_smiles: str = ''
    polymer_id: Optional[str] = None

    # polyBERT predictions
    tg_polybert: Optional[float] = None
    tg_polybert_std: Optional[float] = None

    # Embedding vector (for downstream tasks)
    embedding_dim: int = 0
    embedding_available: bool = False

    # Metadata
    prediction_engine: str = 'polyBERT (DeBERTa)'
    model_name: str = 'kuelumbus/polyBERT'
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'success': self.success,
            'polybert_engine': self.prediction_engine,
            'polybert_model': self.model_name,
            'polybert_embedding_dim': self.embedding_dim,
            'polybert_embedding_available': self.embedding_available,
        }
        if self.tg_polybert is not None:
            d['tg_polybert'] = round(float(self.tg_polybert), 2)
        if self.tg_polybert_std is not None:
            d['tg_polybert_std'] = round(float(self.tg_polybert_std), 3)
        if self.warnings:
            d['polybert_warnings'] = self.warnings
        return d


class PolyBERTPredictor:
    """
    Generate polymer embeddings using polyBERT and predict properties.

    Two-stage pipeline:
    1. polyBERT tokenizer + model → 600-D CLS embedding
    2. Trained sklearn heads (RF on embeddings) → property predictions

    The polyBERT model (~2GB) is loaded lazily on first prediction.
    Prediction heads are trained separately after embedding extraction.
    """

    MODEL_NAME = 'kuelumbus/polyBERT'
    # HuggingFace revision pinning for reproducibility. The polyBERT v1
    # release (Nature Comms, 2023) is what every published Tg/etc. head was
    # trained against; downloading a newer commit at inference time would
    # silently shift our embedding space and invalidate the trained heads.
    #
    # Pin order:
    #   1. ``POLYBERT_REVISION`` env var (ops/CI override)
    #   2. ``MODEL_REVISION_DEFAULT`` class attribute (committed pin)
    # Once the team has run inference on a known-good commit, capture the
    # commit hash this method logs ("polyBERT loaded ... commit=<sha>") and
    # set ``MODEL_REVISION_DEFAULT`` to that sha string. Until then, leaving
    # ``'main'`` is acceptable but means HuggingFace can update the upstream
    # weights underneath us.
    MODEL_REVISION_DEFAULT = 'main'

    @classmethod
    def _resolve_revision(cls) -> str:
        return (os.environ.get('POLYBERT_REVISION') or '').strip() or cls.MODEL_REVISION_DEFAULT

    def __init__(self, model_dir: Optional[str] = None):
        self._tokenizer = None
        self._model = None
        self._prediction_heads = {}   # {prop: sklearn model}
        self._loaded = False

        if model_dir is None:
            from django.conf import settings
            model_dir = str(
                Path(settings.BASE_DIR) / 'poly_x' / 'trained_models' / 'polybert')
        self._model_dir = model_dir

        # Try to load prediction heads (lightweight, load eagerly)
        self._load_prediction_heads()

    def _load_prediction_heads(self):
        """Load trained ensemble prediction heads from model directory.

        Only loads ensemble heads ({prop}_head.joblib), not individual
        RF/GB heads ({prop}_rf_head.joblib, {prop}_gb_head.joblib) which
        are stored for reproducibility but not used at inference time.
        """
        heads_dir = os.path.join(self._model_dir, 'heads')
        if not os.path.isdir(heads_dir):
            return

        # Only load ensemble heads matching known property names
        known_props = ['tg']  # Extend as more properties are trained

        try:
            from core_utils.safe_pickle import safe_joblib_load
            for prop in known_props:
                head_path = os.path.join(heads_dir, f'{prop}_head.joblib')
                if os.path.exists(head_path):
                    self._prediction_heads[prop] = safe_joblib_load(head_path)
                    logger.info("Loaded polyBERT ensemble head: %s (%.1f MB)",
                                prop, os.path.getsize(head_path) / 1e6)
        except Exception as e:
            logger.warning("Failed to load polyBERT prediction heads: %s", e)

    def _load_model(self):
        """Lazily load polyBERT model and tokenizer."""
        if self._loaded:
            return

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library required for polyBERT")

        revision = self._resolve_revision()
        try:
            logger.info("Loading polyBERT model: %s (revision=%s) ...",
                        self.MODEL_NAME, revision)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME, revision=revision)
            self._model = AutoModel.from_pretrained(
                self.MODEL_NAME, revision=revision)
            self._model.eval()

            # Log the actual commit hash HuggingFace resolved to. Capture
            # this in deployment to pin MODEL_REVISION_DEFAULT for full
            # reproducibility - "main" can drift between deploys.
            resolved_sha = (
                getattr(self._model.config, '_commit_hash', None)
                or getattr(self._tokenizer, '_commit_hash', None)
                or 'unknown'
            )
            logger.info(
                "polyBERT resolved revision='%s' commit=%s "
                "(set POLYBERT_REVISION env to pin)",
                revision, resolved_sha,
            )
            self._loaded_revision = revision
            self._loaded_commit = resolved_sha

            # Set deterministic seeds for reproducibility
            torch.manual_seed(42)
            np.random.seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)
                # Ensure deterministic cuDNN operations
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
                logger.info("polyBERT loaded on GPU (deterministic mode)")
            else:
                logger.info("polyBERT loaded on CPU")

            self._loaded = True
        except Exception as e:
            logger.warning("Failed to load polyBERT: %s", e)
            raise

    def is_available(self) -> bool:
        """True if transformers is installed and prediction heads are loaded."""
        return TRANSFORMERS_AVAILABLE and len(self._prediction_heads) > 0

    def get_embedding(self, polymer_smiles: str) -> Optional[np.ndarray]:
        """
        Generate 600-D embedding for a polymer SMILES.

        Uses CLS token output from the last hidden state.
        """
        self._load_model()

        try:
            # polyBERT expects PSMILES with [*] replaced by special tokens
            # The tokenizer handles this natively
            inputs = self._tokenizer(
                polymer_smiles, return_tensors='pt',
                padding=True, truncation=True, max_length=512)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # CLS token embedding (first token of last hidden state)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            flat = cls_embedding.flatten()

            # Free GPU tensors promptly to prevent CUDA memory accumulation
            del inputs, outputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # polyBERT produces 600-D embeddings; assert for reproducibility
            if flat.shape[0] != 600:
                logger.warning("polyBERT embedding dim=%d, expected 600", flat.shape[0])
            return flat

        except Exception as e:
            logger.warning("polyBERT embedding error: %s", e)
            return None

    def predict(self, polymer_smiles: str,
                polymer_id: Optional[str] = None) -> PolyBERTPrediction:
        """
        Predict polymer properties using polyBERT embeddings.

        Steps:
        1. Generate CLS embedding from polyBERT
        2. Pass embedding through trained prediction heads
        """
        result = PolyBERTPrediction(
            polymer_smiles=polymer_smiles,
            polymer_id=polymer_id,
        )

        try:
            embedding = self.get_embedding(polymer_smiles)

            if embedding is None:
                result.success = False
                result.warnings.append('Failed to generate embedding')
                return result

            result.embedding_dim = len(embedding)
            result.embedding_available = True

            # Apply prediction heads
            emb_2d = embedding.reshape(1, -1)
            for prop, head in self._prediction_heads.items():
                try:
                    # Use predict_with_std if available (EnsembleHead)
                    if hasattr(head, 'predict_with_std'):
                        pred, std = head.predict_with_std(emb_2d)
                        pred = float(pred[0])
                        std = float(std[0])
                        if hasattr(result, f'{prop}_polybert'):
                            setattr(result, f'{prop}_polybert', pred)
                        if hasattr(result, f'{prop}_polybert_std'):
                            setattr(result, f'{prop}_polybert_std', std)
                    else:
                        pred = float(head.predict(emb_2d)[0])
                        if hasattr(result, f'{prop}_polybert'):
                            setattr(result, f'{prop}_polybert', pred)
                except Exception as e:
                    logger.warning("polyBERT head prediction failed for %s: %s",
                                   prop, e)

            if not self._prediction_heads:
                result.warnings.append(
                    'No prediction heads trained yet - embedding generated only')

        except ImportError:
            result.success = False
            result.warnings.append('transformers library not installed')
        except Exception as e:
            logger.warning("polyBERT prediction error: %s", e)
            result.success = False
            result.warnings.append(str(e))

        return result
