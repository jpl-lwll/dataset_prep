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
from dataclasses import dataclass, field
from typing import List
from pathlib import Path
import shutil
import pandas as pd
from lwll_dataset_prep.logger import log

@dataclass
class wikimatrix(BaseProcesser):
    """
    The WikiMatrix datasets have 654618, 111469, 298264 pairs that have BLEU >= 1.04 for en-pl, en-si, and en-fa, respectively.
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'wikimatrix'
    _task_type: str = 'machine_translation'
    _urls: List[str] = field(default_factory=lambda: ['https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-pl.tsv.gz',
                                                      'https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-si.tsv.gz',
                                                      'https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-fa.tsv.gz'])
    _sample_percent: float = 0.35
    _test_size: int = 2000

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]

    def download(self) -> None:
        # Download
        # This datasets is from Tim Allison from LFT.
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar files
        # for fname in self._fnames:
        #     self.extract_gz(dir_name=self._path_name, fname=fname)

        for lng in ['pol', 'sin', 'fas']:
            self._path_name_tmp = f"{self._path_name}-{lng}"

            df_train = pd.read_csv(f"{self.path}/{lng}/{lng}-adapt-train-wikimatrix.csv", sep='\t', header=None,
                                   names=[lng, 'english'], error_bad_lines=True)

            df_test = pd.read_csv(f"{self.path}/{lng}/{lng}-adapt-test-wikimatrix.csv", sep='\t', header=None,
                                  names=[lng, 'english'], error_bad_lines=True)

            df_train['target_size'] = df_train['english'].apply(lambda x: len(x))
            df_train.reset_index(inplace=True)
            df_train.rename({'index': 'id'}, inplace=True, axis=1)
            df_train.rename({'english': 'target'}, inplace=True, axis=1)
            df_train.rename({lng: 'source'}, inplace=True, axis=1)

            size = len(df_train)
            df_test['target_size'] = df_test['english'].apply(lambda x: len(x))
            df_test.reset_index(inplace=True)
            df_test.rename({'index': 'id'}, inplace=True, axis=1)
            df_test.rename({'english': 'target'}, inplace=True, axis=1)
            df_test.rename({lng: 'source'}, inplace=True, axis=1)
            df_test['id'] = df_test['id'] + size

            # Create Sample vs. Full and Train vs. Test Splits
            log.info('Generating splits...')
            train_full = df_train
            test_full = df_test
            train_sample = train_full.iloc[:int(len(train_full) * self._sample_percent)]
            test_sample = test_full.iloc[0:len(test_full)]

            # Verify id data type as str
            train_full['id'] = train_full['id'].astype(str)
            test_full['id'] = test_full['id'].astype(str)
            train_sample['id'] = train_sample['id'].astype(str)
            test_sample['id'] = test_sample['id'].astype(str)

            log.info(f"Dataset splits output:")
            log.info(f"train_full: {len(train_full)}")
            log.info(f"test_full: {len(test_full)}")
            log.info(f"train_sample: {len(train_sample)}")
            log.info(f"test_sample: {len(test_sample)}")

            sample_total_codecs = int(train_sample['target_size'].sum())
            full_total_codecs = int(train_full['target_size'].sum())

            # Resetting defualt indices for feather saving
            train_full.reset_index(inplace=True, drop=True)
            test_full.reset_index(inplace=True, drop=True)
            train_sample.reset_index(inplace=True, drop=True)
            test_sample.reset_index(inplace=True, drop=True)

            # Set up paths
            log.info('Creating Local Paths...')
            sample_path = str(self.data_path.joinpath(f"{self._path_name_tmp}").joinpath(f'{self._path_name_tmp}_sample'))
            full_path = str(self.data_path.joinpath(f"{self._path_name_tmp}").joinpath(f'{self._path_name_tmp}_full'))
            sample_labels_path = str(self.labels_path.joinpath(f"{self._path_name_tmp}").joinpath(f'{self._path_name_tmp}_sample'))
            full_labels_path = str(self.labels_path.joinpath(f"{self._path_name_tmp}").joinpath(f'{self._path_name_tmp}_full'))
            Path(sample_path).mkdir(exist_ok=True, parents=True)
            Path(full_path).mkdir(exist_ok=True, parents=True)
            Path(sample_labels_path).mkdir(exist_ok=True, parents=True)
            Path(full_labels_path).mkdir(exist_ok=True, parents=True)

            # Save our to paths
            log.info('Saving processed dataset out...')
            train_sample[['id', 'source']].to_feather(f"{sample_path}/train_data.feather")
            train_full[['id', 'source']].to_feather(f"{full_path}/train_data.feather")
            test_sample[['id', 'source']].to_feather(f"{sample_path}/test_data.feather")
            test_full[['id', 'source']].to_feather(f"{full_path}/test_data.feather")

            train_sample[['id', 'target']].to_feather(f"{sample_labels_path}/labels_train.feather")
            train_full[['id', 'target']].to_feather(f"{full_labels_path}/labels_train.feather")
            test_sample[['id', 'target']].to_feather(f"{sample_labels_path}/labels_test.feather")
            test_full[['id', 'target']].to_feather(f"{full_labels_path}/labels_test.feather")

            test_sample[['id']].to_feather(f"{sample_labels_path}/test_label_ids.feather")
            test_full[['id']].to_feather(f"{full_labels_path}/test_label_ids.feather")

            dataset_doc = DatasetDoc(name=f"{self._path_name_tmp}",
                                     dataset_type='machine_translation',
                                     sample_number_of_samples_train=len(train_sample),
                                     sample_number_of_samples_test=len(test_sample),
                                     sample_number_of_classes=None,
                                     full_number_of_samples_train=len(train_full),
                                     full_number_of_samples_test=len(test_full),
                                     full_number_of_classes=None,
                                     number_of_channels=None,
                                     classes=None,
                                     language_from=lng,
                                     language_to='eng',
                                     sample_total_codecs=sample_total_codecs,
                                     full_total_codecs=full_total_codecs,
                                     license_link='https://github.com/facebookresearch/LASER/tree/master/tasks/WikiMatrix',
                                     license_requirements='Creative Commons Attribution-ShareAlike license',
                                     license_citation=' Holger Schwenk, Vishrav Chaudhary, Shuo Sun, Hongyu Gong and Paco Guzman,\
                                     WikiMatrix: Mining 135M Parallel Sentences in 1620 Language Pairs from Wikipedia arXiv, July 11 2019.',
                                     )
            log.info('Saving Metadata...')
            self.save_dataset_metadata(dir_name=f"{self._path_name_tmp}", metadata=dataset_doc)

            # Validate sample subsets
            log.info("Validating sample datasets")
            self.validate_mt(name=self._path_name_tmp,
                             full_path=full_path,
                             sample_path=sample_path,
                             full_labels_path=full_labels_path,
                             sample_labels_path=sample_labels_path,
                             metadata=dataset_doc,
                             task_type=self._task_type)

            # Clean up unecessary files in directory for tar
            log.info(f"{self.path}/{lng}")
            shutil.rmtree(f"{self.path}/{lng}")

            # Push data to DynamoDB
            log.info('Pushing labels to Dynamo')
            self.push_to_dynamo(pd.concat([train_full, test_full]), f"wikimatrix-{lng}_full", 'id', 'target', 'target_size')
            self.push_to_dynamo(pd.concat([train_sample, test_sample]), f"wikimatrix-{lng}_sample", 'id', 'target', 'target_size')
        return

    def transfer(self) -> None:
        for lng in ['pol', 'sin', 'fas']:
            log.info(f"Pushing artifacts to appropriate cloud resources for {self._path_name}-{lng}...")
            name = str(f"{self._path_name}-{lng}")
            self.push_data_to_cloud(dir_name=name, dataset_type='development', task_type=self._task_type, is_mt=True)
            log.info("Done")
        return
