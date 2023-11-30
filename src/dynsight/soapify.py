"""SOAPify package."""

from dynsight._internal.soapify.saponify import (
    saponify_multiple_trajectories,
    saponify_trajectory,
)
from dynsight._internal.soapify.utilities import (
    fill_soap_vector_from_dscribe,
    get_soap_settings,
)

__all__ = [
    "saponify_trajectory",
    "saponify_multiple_trajectories",
    "get_soap_settings",
    "fill_soap_vector_from_dscribe",
]
