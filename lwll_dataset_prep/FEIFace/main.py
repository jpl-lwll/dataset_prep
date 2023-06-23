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
import glob
import os
from typing import List
from pathlib import Path
import pandas as pd
from lwll_dataset_prep.logger import log


@dataclass
class FEIFace(BaseProcesser):
    """
    Our source data is in the form:
    originalimages_part1
    - 1-01.jpg
    - 1-02.jpg
    - ...
    - 1-14.jpg
    - ...
    - 50-01.jpg
    - ...
    originalimages_part2
    - 51-01.jpg
        ...
    - 100-14.jpg


    2800 images from 200 individuals. Planned to use 01 and 02 (left profile) and 09 and 10 (right profile) as testing images.

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'FEIFace'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['https://fei.edu.br/~cet/originalimages_part1.zip',
                                                      'https://fei.edu.br/~cet/originalimages_part2.zip',
                                                      'https://fei.edu.br/~cet/originalimages_part3.zip',
                                                      'https://fei.edu.br/~cet/originalimages_part4.zip'])
    _sample_size_train: int = 8
    _sample_size_test: int = 100
    _k_seed: List[int] = field(default_factory=lambda: [8])
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])

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
            self.extract_zip(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        training_ids = []
        training_class = []
        testing_ids = []
        testing_class = []

        side_profiles = ["01", "02", "09", "10"]

        imgs = sorted(glob.glob(f"{self.path}/*.jpg"))

        pth_to_id = {}
        id_to_pth = {}

        cnt = 0
        for img in imgs:
            cnt += 1
            base = os.path.basename(img)
            new_id = f'img_{cnt}.jpg'
            pth_to_id[base] = new_id
            id_to_pth[new_id] = base
            name = os.path.splitext(base)[0]
            cate, profile = name.split('-')
            if profile in side_profiles:
                testing_ids.append(new_id)
                testing_class.append(cate)
            else:
                training_ids.append(new_id)
                training_class.append(cate)

        df_train = pd.DataFrame({'id': training_ids, 'class': training_class})
        df_test = pd.DataFrame({'id': testing_ids, 'class': testing_class})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)

        # Move the raw data files
        orig_train_paths = [self.path.joinpath(id_to_pth[_id]) for _id in df_train['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           new_names=df_train['id'].unique().tolist())

        orig_test_paths = [self.path.joinpath(id_to_pth[_id]) for _id in df_test['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test',
                                           new_names=df_test['id'].unique().tolist())

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=len(df_train_sample),
                                 sample_number_of_samples_test=len(df_test_sample),
                                 sample_number_of_classes=len(sorted(list(set(training_class)))),
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=len(sorted(list(set(training_class)))),
                                 number_of_channels=1,
                                 classes=sorted(list(set(training_class))),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://fei.edu.br/~cet/facedatabase.html',
                                 license_requirements='None',
                                 license_citation='C. E. Thomaz and G. A. Giraldi. A new ranking method for Principal Components Analysis and its application to face image analysis, Image and Vision Computing, vol. 28, no. 6, pp. 902-913, June 2010',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=list(set(training_class)),
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup downloaded zip files
        log.info("Cleaning up extracted zip copy..")
        for f in self._fnames:
            os.remove(self.path.joinpath(f))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
