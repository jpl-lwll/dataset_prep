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

from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass

from pathlib import Path
import os
import pandas as pd
import numpy as np
from lwll_dataset_prep.logger import log

@dataclass
class cc_aligned_polish(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'cc_aligned_polish'
    _task_type: str = 'machine_translation'
    _sample_percent: float = 0.35
    _test_size: int = 2000

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)

    def download(self) -> None:
        # TODO: need to write a new function under 'process_interface.py' to download from s3 given new rules
        pass

    def process(self) -> None:

        log.info(f"Read in files...")
        df_train = pd.read_csv(f"{self.path}/pol-base-train-cc.csv", sep='\t', header=None,
                               names=['source', 'target'], error_bad_lines=False)
        df_test = pd.read_csv(f"{self.path}/pol-base-test-cc.csv", sep='\t', header=None,
                              names=['source', 'target'], error_bad_lines=False)

        df_train.drop_duplicates(inplace=True)
        df_test.drop_duplicates(inplace=True)
        df_train = df_train.dropna()
        df_test = df_test.dropna()

        df_train['id'] = list(np.arange(df_train.shape[0]))
        df_train['target_size'] = df_train['target'].apply(lambda x: len(x))
        len_train = df_train.shape[0]
        df_test['id'] = list(np.arange(len_train, len_train+df_test.shape[0]))
        df_test['target_size'] = df_test['target'].apply(lambda x: len(x))

        # id and target size columns need to be strings for feather step
        df_train[['source', 'target', 'id']] = df_train[['source', 'target', 'id']].astype('str')
        df_test[['source', 'target', 'id']] = df_test[['source', 'target', 'id']].astype('str')
        log.info(f"{df_test.head()}")

        log.info('Creating sample...')
        df_train_sample = df_train.iloc[:int(df_train.shape[0] * self._sample_percent)]
        df_test_sample = df_test.iloc[0:df_test.shape[0]]

        log.info(f"Dataset size")
        log.info(f"train_full: {df_train.shape[0]}")
        log.info(f"test_full: {df_test.shape[0]}")
        log.info(f"train_sample: {df_train_sample.shape[0]}")
        log.info(f"test_sample: {df_test_sample.shape[0]}")

        sample_total_codecs = int(df_train_sample['target_size'].sum())
        full_total_codecs = int(df_train['target_size'].sum())

        # Resetting default indices for feather saving
        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_train_sample.reset_index(inplace=True, drop=True)
        df_test_sample.reset_index(inplace=True, drop=True)

        # Set up paths
        log.info('Creating Local Paths...')
        sample_path = str(self.data_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_sample'))
        full_path = str(self.data_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_full'))
        sample_labels_path = str(self.labels_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_sample'))
        full_labels_path = str(self.labels_path.joinpath(f"{self._path_name}").joinpath(f'{self._path_name}_full'))

        Path(sample_path).mkdir(exist_ok=True, parents=True)
        Path(full_path).mkdir(exist_ok=True, parents=True)
        Path(sample_labels_path).mkdir(exist_ok=True, parents=True)
        Path(full_labels_path).mkdir(exist_ok=True, parents=True)

        # Save our to paths
        log.info('Saving processed dataset out...')
        df_train_sample[['id', 'source']].to_feather(f"{sample_path}/train_data.feather")
        df_train[['id', 'source']].to_feather(f"{full_path}/train_data.feather")
        df_test_sample[['id', 'source']].to_feather(f"{sample_path}/test_data.feather")
        df_test[['id', 'source']].to_feather(f"{full_path}/test_data.feather")

        df_train_sample[['id', 'target']].to_feather(f"{sample_labels_path}/labels_train.feather")
        df_train[['id', 'target']].to_feather(f"{full_labels_path}/labels_train.feather")
        df_test_sample[['id', 'target']].to_feather(f"{sample_labels_path}/labels_test.feather")
        df_test[['id', 'target']].to_feather(f"{full_labels_path}/labels_test.feather")

        df_test_sample[['id']].to_feather(f"{sample_labels_path}/test_label_ids.feather")
        df_test[['id']].to_feather(f"{full_labels_path}/test_label_ids.feather")

        dataset_doc = DatasetDoc(name=f"{self._path_name}",
                                 dataset_type='machine_translation',
                                 sample_number_of_samples_train=df_train_sample.shape[0],
                                 sample_number_of_samples_test=df_test_sample.shape[0],
                                 sample_number_of_classes=None,
                                 full_number_of_samples_train=df_train.shape[0],
                                 full_number_of_samples_test=df_test.shape[0],
                                 full_number_of_classes=None,
                                 number_of_channels=None,
                                 classes=None,
                                 language_from='pol',
                                 language_to='eng',
                                 sample_total_codecs=sample_total_codecs,
                                 full_total_codecs=full_total_codecs,
                                 license_link='http://www.statmt.org/cc-aligned/',
                                 license_requirements='None',
                                 license_citation="@inproceedings{elkishky_ccaligned_2020, author = {El-Kishky, Ahmed and Chaudhary, Vishrav and \
                                 Guzm{'a}n, Francisco and Koehn, Philipp}, booktitle = {Proceedings of the 2020 Conference on Empirical Methods in \
                                 Natural Language Processing (EMNLP 2020)},month = {November}, title = {CCAligned: A Massive Collection of \
                                 Cross-lingual Web-Document Pairs}, year = {2020}, address = Online, publisher = Association for Computational \
                                 Linguistics, url = https://www.aclweb.org/anthology/2020.emnlp-main.480, doi = 10.18653/v1/2020.emnlp-main.480, \
                                 pages = 5960--5969}'",
                                 )
        log.info('Saving Metadata...')
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Clean up unecessary files in directory for tar
        os.remove(f"{self.path}/pol-base-train-cc.csv")
        os.remove(f"{self.path}/pol-base-test-cc.csv")

        # Push data to DynamoDB
        log.info('Pushing labels to Dynamo')
        self.push_to_dynamo(pd.concat([df_train, df_test]), 'cc_aligned_polish_full', 'id', 'target', 'target_size')
        self.push_to_dynamo(pd.concat([df_train_sample, df_test_sample]), 'cc_aligned_polish_sample', 'id', 'target', 'target_size')
        return

    def transfer(self) -> None:
        log.info(f"Pushing artifacts to appropriate cloud resources for {self._path_name}...")
        name = str(f"{self._path_name}")
        self.push_data_to_cloud(dir_name=name, dataset_type='development', task_type=self._task_type, is_mt=True)
        log.info("Done")
        return
