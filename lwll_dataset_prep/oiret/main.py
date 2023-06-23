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
import os
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log


@dataclass
class oiret(BaseProcesser):
    """
    This data set comes form Ben Johnson, there are 144 training images and 55,428 test images.
    There are 9 categories: ['submarine', 'helicopter', 'welding', 'bulldozer', 'wrench', 'missile', 'sailor', 'miner', 'soldier']

    The labels files from the tar requires a newer version of pyarrow, so I manually convert it to CSV.
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'oiret'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['https://s3.amazonaws.com/jataware.jata.lol/data/oiret.tar'])
    _sample_size_train: int = 5000
    _sample_size_test: int = 1000
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])
    _k_seed: List[int] = field(default_factory=lambda: [1])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        # Create our data schema
        df_train = pd.read_csv(f"{self.path}/labels_train.csv")
        df_test = pd.read_csv(f"{self.path}/labels_test.csv")

        classes = list(set(df_test['class'].tolist()))

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)
        # Move the raw data files
        orig_train_paths = [_p for _p in self.path.joinpath(f'oiret/oiret/oiret_full/train').glob('*') if _p.suffix in self._valid_extensions]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train', delete_original=False)

        orig_test_paths = [_p for _p in self.path.joinpath(f'oiret/oiret/oiret_full/test').glob('*') if _p.suffix in self._valid_extensions]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test', delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=len(df_train_sample),
                                 sample_number_of_samples_test=len(df_test_sample),
                                 sample_number_of_classes=10,
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=10,
                                 number_of_channels=1,
                                 classes=classes,
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='None',
                                 license_requirements='None',
                                 license_citation='None',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=classes,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        shutil.rmtree(self.path.joinpath("oiret"))
        os.remove(self.path.joinpath("oiret.tar"))
        os.remove(self.path.joinpath("labels_train.csv"))
        os.remove(self.path.joinpath("labels_test.csv"))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")
