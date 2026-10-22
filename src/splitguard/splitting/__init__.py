"""Split-strategy recommendation and explicit split creation."""

from splitguard.splitting.recommendations import (
    SplitConfirmationError,
    SplitResult,
    SplitStrategy,
    create_split,
    recommend_split,
)

__all__ = [
    "SplitConfirmationError",
    "SplitResult",
    "SplitStrategy",
    "create_split",
    "recommend_split",
]
