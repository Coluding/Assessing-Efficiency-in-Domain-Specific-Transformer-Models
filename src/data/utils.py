from dataclasses import dataclass
from pathlib import Path
from typing import *
import yaml

@dataclass
class DatasetInfo:
    """
        Class containing meta information about dataset
    """
    name: str
    subset: Optional[str]
    text_columns: List[str]
    validation_set_names: List[str]
    test_set_names: List[str]
    sentence_segmentation: bool
    num_clf_classes: int  # Number of classification labels
    num_regr: int  # Number of regression labels
    save_local_path: Optional[Path] = None

    @property
    def is_classification(self):
        """
        :return: True if classification task else False
        """
        return self.num_clf_classes > 0

    @property
    def is_regression(self):
        """
        :return: True if regression task else False
        """
        return not self.is_classification

    @property
    def is_downstream(self):
        """
        :return: True if downstream task else False
        """
        return (self.num_regr > 0) or (self.num_clf_classes > 0)

    @property
    def is_pretraining(self):
        """
        :return: True if pretraining task else False
        """
        return not self.is_downstream

    def __post_init(self):
        if self.is_downstream:
            assert (self.num_clf_classes == 0) or (self.num_regr == 0), "Only single task are allowed"


def dataset_info_from_yaml(yaml_path: str) -> List[DatasetInfo]:
    """
        Load DatasetInfo from yaml file
    :param yaml_path:
    :return:
    """
    with open(yaml_path, "r") as f:
        dataset_info_dict = yaml.safe_load(f)

    return [DatasetInfo(**x) for x in dataset_info_dict]


_dataset_infos = dataset_info_from_yaml(yaml_path="../data/dataset_infos.yaml")

def get_dataset_info(dataset_name: str, dataset_subset: str) -> DatasetInfo:
    """
        Return the DatasetInfo based on name and subset
    :param dataset_name:
    :param dataset_subset:
    :return:
    """

    dataset_infos = []
    for _dataset_info in _dataset_infos:
        if (_dataset_info.name == dataset_name) and ((_dataset_info.subset is None and dataset_subset is None)
                                                     or (_dataset_info.subset == dataset_subset)):
            dataset_infos += [_dataset_info]

    assert len(dataset_infos) == 1, dataset_infos
    return dataset_infos[0]