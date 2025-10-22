from typing import Dict, Type

from flow_matching.datasets.synthetic_datasets import (
    DatasetCheckerboard,
    DatasetInvertocat,
    DatasetMixture,
    DatasetMoons,
    DatasetSiggraph,
    SyntheticDataset,
)

TOY_DATASETS: Dict[str, Type[SyntheticDataset]] = {
    "moons": DatasetMoons,
    "mixture": DatasetMixture,
    "siggraph": DatasetSiggraph,
    "checkerboard": DatasetCheckerboard,
    "invertocat": DatasetInvertocat,
}
