"""Model-family adapters.

Importing this package registers built-in adapters without loading models.
"""

from .flux2_klein_9b import Flux2Klein9BAdapter
from .z_image_turbo import ZImageTurboAdapter

__all__ = ["Flux2Klein9BAdapter", "ZImageTurboAdapter"]
