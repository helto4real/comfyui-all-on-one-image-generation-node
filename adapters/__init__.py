"""Model-family adapters.

Importing this package registers built-in adapters without loading models.
"""

from .flux2_klein_9b import Flux2Klein9BAdapter
from .ideogram4 import Ideogram4Adapter
from .krea2 import Krea2Adapter
from .z_image_turbo import ZImageTurboAdapter

__all__ = ["Flux2Klein9BAdapter", "Ideogram4Adapter", "Krea2Adapter", "ZImageTurboAdapter"]
