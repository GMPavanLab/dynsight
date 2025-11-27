"""data processing package."""

from dynsight._internal.data_processing.auto_filtering import (
    AutoFilteringPipeline,
    auto_filtering,
)
from dynsight._internal.data_processing.classify import (
    applyclassification,
    createreferencesfromtrajectory,
    getdistancebetween,
    getdistancesfromref,
    getdistancesfromrefnormalized,
    getreferencesfromdataset,
    mergereferences,
    savereferences,
)
from dynsight._internal.data_processing.distances import (
    kernelsoap,
    simplekernelsoap,
    simplesoapdistance,
    soapdistance,
    soapdistancenormalized,
)

__all__ = [
    "AutoFilteringPipeline",
    "applyclassification",
    "auto_filtering",
    "createreferencesfromtrajectory",
    "getdistancebetween",
    "getdistancesfromref",
    "getdistancesfromrefnormalized",
    "getreferencesfromdataset",
    "kernelsoap",
    "mergereferences",
    "savereferences",
    "simplekernelsoap",
    "simplesoapdistance",
    "soapdistance",
    "soapdistancenormalized",
]
