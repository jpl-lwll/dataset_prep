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
import glob
import os
import pathlib
from pathlib import Path
import pandas as pd
import random
import shutil
from lwll_dataset_prep.logger import log

@dataclass
class AID(BaseProcesser):
    """
    Source data: https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ
    10,000 images in 30 categories

    Folder structure:
    Airport/airport_1.jpg
    Airport/airport_10.jpg
    Airport/airport_100.jpg
    Airport/airport_101.jpg
    ...

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'AID'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['AID.zip'])
    _sample_size_train: int = 500
    _sample_size_test: int = 500
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        log.info("we have to sign in and download this dataset manually: https://1drv.ms/u/s!AthY3vMZmuxChNR0Co7QHpJ56M-SvQ")
        log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_zip(dir_name=self._path_name, fname=fname)

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        classes = [os.path.basename(c) for c in glob.glob(os.path.join(self.path, "AID/*"))]

        orig_paths = []
        cnt = 0
        for _img in glob.glob(os.path.join(self.path, "AID/*/*")):
            orig_paths.append(pathlib.Path(_img))

        random.Random(5).shuffle(orig_paths)
        split_idx = int(len(orig_paths) * 0.7)
        orig_train_paths = orig_paths[:split_idx]
        orig_test_paths = orig_paths[split_idx:]

        training_ids = []
        testing_ids = []

        pth_to_id = {}
        id_to_pth = {}
        cnt = 0

        for p in orig_train_paths:
            cnt += 1
            new_id = f'img_{cnt}.jpg'
            pth_to_id[p] = new_id
            id_to_pth[new_id] = p
            training_ids.append(new_id)

        for p in orig_test_paths:
            cnt += 1
            new_id = f'img_{cnt}.jpg'
            pth_to_id[p] = new_id
            id_to_pth[new_id] = p
            testing_ids.append(new_id)

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'class': [os.path.basename(os.path.dirname(id_to_pth[i])) for i in training_ids]})
        df_test = pd.DataFrame({'id': testing_ids, 'class': [os.path.basename(os.path.dirname(id_to_pth[i])) for i in testing_ids]})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=False)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           delete_original=False, new_names=df_train['id'].unique().tolist())

        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test',
                                           delete_original=False, new_names=df_test['id'].unique().tolist())

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type=self._task_type,
                                 sample_number_of_samples_train=len(df_train_sample["id"].unique()),
                                 sample_number_of_samples_test=len(df_test_sample["id"].unique()),
                                 sample_number_of_classes=len(classes),
                                 full_number_of_samples_train=len(df_train["id"].unique()),
                                 full_number_of_samples_test=len(df_test["id"].unique()),
                                 full_number_of_classes=len(classes),
                                 number_of_channels=3,
                                 classes=classes,
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='None',
                                 license_requirements='None',
                                 license_citation='Xia, Gui-Song, Jingwen Hu, Fan Hu, Baoguang Shi, Xiang Bai, Yanfei Zhong, Liangpei Zhang, and Xiaoqiang Lu. "AID: A benchmark data set for performance evaluation of aerial scene classification." IEEE Transactions on Geoscience and Remote Sensing 55, no. 7 (2017): 3965-3981.',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validate sample subsets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=classes,
                                       train_df=df_train, test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        os.remove(self.path.joinpath("AID.zip"))
        shutil.rmtree(self.path.joinpath("AID"))

        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
