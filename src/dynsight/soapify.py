"""SOAPify package."""

from dynsight._internal.soapify.saponify import (
    fill_soap_vector_from_dscribe,
    saponify_trajectory,
)

__all__ = [
    "fill_soap_vector_from_dscribe",
    "saponify_trajectory",
]
