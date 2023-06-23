# Copyright (c) 2023 California Institute of Technology (“Caltech”). U.S.
# Government sponsorship acknowledged.
# All rights reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of Caltech nor its operating division, the Jet Propulsion
#   Laboratory, nor the names of its contributors may be used to endorse or
#   promote products derived from this software without specific prior written
#   permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

from dataclasses import dataclass
from typing import Optional, List, Dict


# METRICS = ['accuracy']
DATASET_TYPES = ['image_classification', 'object_detection', 'machine_translation', 'video_classification']

# @dataclass
# class ProblemDoc:
#     problem_id: str
#     base_dataset: str
#     base_label_budget: int
#     base_evaluation_metrics: List[str]
#     adaptation_dataset: str
#     adaptation_label_budget: int
#     adaptation_evalutation_metrics: List[str]
#     base_can_use_pretrained_model: bool = True
#     adaptation_can_use_pretrained_model: bool = True
#     blacklisted_resources: List[str] = field(default_factory=lambda: [])

#     def __post_init__(self) -> None:
#         self._validate_metrics(self.base_evaluation_metrics)
#         self._validate_metrics(self.adaptation_evalutation_metrics)

#     @staticmethod
#     def _validate_metrics(metrics: List[str]) -> None:
#         for metric in metrics:
#             if metric not in METRICS:
#                 raise Exception(f'Metric: {metric} not supported yet.')

@dataclass
class DatasetDoc:
    name: str
    dataset_type: str
    sample_number_of_samples_train: int
    sample_number_of_samples_test: int
    sample_number_of_classes: Optional[int]  # Only in image problems
    full_number_of_samples_train: int
    full_number_of_samples_test: int
    full_number_of_classes: Optional[int]  # Only in image problems
    number_of_channels: Optional[int]  # Only in image problems
    classes: Optional[List[str]]
    language_to: Optional[str]  # Only in MT problems
    language_from: Optional[str]  # Only in MT problems
    sample_total_codecs: Optional[int]  # Only in MT problems
    full_total_codecs: Optional[int]  # Only in MT problems
    license_requirements: str
    license_link: str
    license_citation: str

    def __post_init__(self) -> None:
        self._validate_dataset_type(self.dataset_type)

    @staticmethod
    def _validate_dataset_type(dataset_type: str) -> None:
        if dataset_type not in DATASET_TYPES:
            raise Exception(f'Dataset Type: {dataset_type} not supported yet.')

@dataclass
class DatasetDoc_zsl(DatasetDoc):
    zsl_description: Dict[str, str]  # Only in ZSL problems
    seen_classes: List[str]
    unseen_classes: List[str]
