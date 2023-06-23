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

import os
import pandas as pd
import subprocess
import sys
import pickle
import shutil
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from lwll_dataset_prep.logger import log


@dataclass
class quick_draw_dataset(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'quick_draw_dataset'
    _task_type: str = 'image_classification'
    _sample_size_train: int = 2000
    _sample_size_test: int = 400
    _k_seed: List[int] = field(default_factory=lambda: [1])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')

        self.classes: List[Any] = []
        self.data: List[Dict[str, Any]] = []

    def download(self) -> None:
        log.info(f"Quick Draw Dataset requires instaling an quick draw package")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'quickdraw'])
        from quickdraw import QuickDrawDataGroup

        class_list = pickle.load(open(os.path.join('lwll_dataset_prep', 'quick_draw_dataset', 'class_list.p'), 'rb'))
        assert (len(class_list) == 345)

        id = 0
        for clss in class_list:
            if not os.path.exists(os.path.join(self.path, 'imgs_save', clss)):
                os.makedirs(os.path.join(self.path, 'imgs_save', clss))
            drawings = QuickDrawDataGroup(clss)
            for im in drawings.drawings:
                im.image.save(os.path.join(self.path, 'imgs_save', clss, f"img{id}.png"))
                id += 1

    def process(self) -> None:

        class_list = pickle.load(open(os.path.join('lwll_dataset_prep', 'quick_draw_dataset', 'class_list.p'), 'rb'))
        class_list_dir = os.listdir(os.path.join('lwll_datasets', 'quick_draw_dataset', 'imgs'))
        if '.DS_Store' in class_list_dir:
            class_list_dir.remove('.DS_Store')
        assert (len(class_list) == len(class_list_dir))

        self.setup_folder_structure(dir_name=self._path_name)

        df_data = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        data: Dict = {'id': [], 'class': [], 'orig_path': []}

        for clss in class_list:
            all_files = os.listdir(os.path.join('lwll_datasets', 'quick_draw_dataset', 'imgs', clss))
            im_list = [im for im in all_files if im.split('.')[-1] == 'png']
            for im in im_list:
                if im.split('.')[-1] == 'png':
                    data['id'].append(im)
                    data['class'].append(clss)
                    data['orig_path'].append(self.path.joinpath('imgs').joinpath(clss).joinpath(im))

        df_data = df_data.from_dict(data)
        df_data = df_data.sample(frac=1, random_state=1).reset_index(drop=True)
        n = df_data.shape[0]
        df_train = df_data.iloc[:int(0.7 * n), :]
        df_test = df_data.iloc[int(0.7 * n):, :]
        overlap = list(set(df_train['id']) & set(df_test['id']))
        assert (len(overlap) == 0)

        # Extract original paths
        orig_paths_train = df_train['orig_path'].tolist()
        orig_paths_test = df_test['orig_path'].tolist()

        df_train = df_train.drop(['orig_path'], axis=1)
        df_test = df_test.drop(['orig_path'], axis=1)

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name,
                                                                  df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name,
                                           orig_paths=orig_paths_train,
                                           dest_type='train',
                                           delete_original=False)
        self.original_paths_to_destination(dir_name=self._path_name,
                                           orig_paths=orig_paths_test,
                                           dest_type='test',
                                           delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name,
                                                  df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                dataset_type='image_classification',
                                sample_number_of_samples_train=len(df_train_sample),
                                sample_number_of_samples_test=len(df_test_sample),
                                sample_number_of_classes=df_train_sample['class'].nunique(),
                                full_number_of_samples_train=len(df_train),
                                full_number_of_samples_test=len(df_test),
                                full_number_of_classes=len(class_list),
                                number_of_channels=1,
                                classes=class_list,
                                language_from=None,
                                language_to=None,
                                sample_total_codecs=None,
                                full_total_codecs=None,
                                license_link='https://github.com/googlecreativelab/quickdraw-dataset',
                                license_requirements='CC 4.0 Attribution Required',
                                license_citation='None')  # noqa
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_paths_train,
                                       orig_test_paths=orig_paths_test,
                                       classes=class_list,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Removing imgs directory
        shutil.rmtree(self.path.joinpath('imgs'))
        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
