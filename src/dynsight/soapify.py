"""SOAPify package."""

from dynsight._internal.soapify.saponify import (
    saponify_trajectory,
)
from dynsight._internal.soapify.utilities import (
    fill_soap_vector_from_dscribe,
    get_soap_settings,
)

__all__ = [
    "fill_soap_vector_from_dscribe",
    "get_soap_settings",
    "saponify_trajectory",
]
