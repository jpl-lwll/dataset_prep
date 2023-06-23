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

import shutil
import os
import pandas as pd
import json
import ntpath
import random
import zipfile
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from lwll_dataset_prep.logger import log


@dataclass
class deep_fashion_2(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'deep_fashion_2'
    _task_type: str = 'object_detection'
    _url: str = 'raw/deep_fashion_2.zip'
    _fname: str = 'deep_fashion_2.zip'
    _sample_size_train: int = 800
    _sample_size_test: int = 300

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')

        self.classes: List[Any] = []
        self.data: List[Dict[str, Any]] = []

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

    def _get_paths(self) -> Tuple[List, List]:

        annot_pt = [self.path.joinpath('annos').joinpath(p)
                    for p in os.listdir(self.path.joinpath('annos'))
                    if os.path.splitext(p)[-1] == '.json']
        img_pt = [self.path.joinpath('image').joinpath(p)
                  for p in os.listdir(self.path.joinpath('image'))
                  if os.path.splitext(p)[-1] == '.jpg']
        return img_pt, annot_pt

    def _read_annotation(self, path: Path) -> Tuple[List, List]:

        bboxes, labels = [], []

        with open(path) as f:
            data = json.load(f)
            del data['source']
            del data['pair_id']

            for item in data.keys():
                bboxes.append(data[item]['bounding_box'])
                labels.append(data[item]['category_name'])

        assert (len(bboxes) == len(labels))

        return bboxes, labels

    def _get_string(self, bbox: List[int]) -> str:
        bbox_string = ""
        for i, c in enumerate(bbox):
            if (i > 0):
                bbox_string += ", "
            bbox_string += str(c)
        return bbox_string

    def _unzip_file(self, filepath: Path, keep_file: bool = False) -> None:
        log.info(f"Unzipping  {filepath}")
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(self.path)

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

        if not keep_file:
            os.remove(filepath)

    def download(self) -> None:
        log.info(f"Downloading dataset {self._path_name}")
        self.download_s3_url(data_path=self._url, dest=self.path.joinpath(self._fname))
        self._unzip_file(self.path.joinpath(self._fname))
        log.info("Done")

    def process(self) -> None:

        log.info(f"Processing dataset {self._path_name}")

        img_paths, lb_paths = self._get_paths()

        for img_pt, l_pt in zip(img_paths, lb_paths):
            bboxes, labels = self._read_annotation(l_pt)

            # Adding labels to structure if not seen before
            for lb in labels:
                if lb not in self.classes:
                    self.classes.append(lb)

            # Adding bounding box information to structure
            for bbox, lb in zip(bboxes, labels):
                self.data.append({'id': ntpath.basename(img_pt),
                                  'bbox': self._get_string(bbox),
                                  'class': lb,
                                  'orig_path': img_pt})

        # Creating dataframe from data
        df = pd.DataFrame(columns=['id', 'label', 'class', 'orig_path']).from_dict(self.data)

        # Get unique image id's and shuffle their order with 'seed=5'
        image_ids = df.id.unique()
        random.Random(5).shuffle(image_ids)

        # Split train/test
        train_imgs = image_ids[:int(0.7*len(image_ids))]
        test_imgs = image_ids[int(0.7*len(image_ids)):]

        # Create training and test dataframes
        df_train = df[df['id'].isin(train_imgs)]
        df_test = df[df['id'].isin(test_imgs)]

        overlap = list(set(df_train['id']) & set(df_test['id']))
        assert (len(overlap) == 0)

        # Extract original paths
        orig_paths_train = df_train['orig_path'].tolist()
        orig_paths_test = df_test['orig_path'].tolist()

        # Convert to Path
        orig_paths_train = [Path(p) for p in orig_paths_train]
        orig_paths_test = [Path(p) for p in orig_paths_test]

        df_train = df_train.drop(['orig_path'], axis=1)
        df_test = df_test.drop(['orig_path'], axis=1)

        # Create sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=f"{self._path_name}",
                                                                  df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test,
                                                                  many_to_one=True)
        # Move the raw files
        self.original_paths_to_destination(dir_name=f"{self._path_name}",
                                           orig_paths=orig_paths_train,
                                           dest_type='train',
                                           delete_original=False)
        self.original_paths_to_destination(dir_name=f"{self._path_name}",
                                           orig_paths=orig_paths_test,
                                           dest_type='test',
                                           delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=f"{self._path_name}",
                                                  df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=f"{self._path_name}",
                                dataset_type='object_detection',
                                sample_number_of_samples_train=len(df_train_sample['id'].unique()),
                                sample_number_of_samples_test=len(df_test_sample['id'].unique()),
                                sample_number_of_classes=len(self.classes),
                                full_number_of_samples_train=len(df_train['id'].unique()),
                                full_number_of_samples_test=len(df_test['id'].unique()),
                                full_number_of_classes=len(self.classes),
                                number_of_channels=3,
                                classes=self.classes,
                                language_from=None,
                                language_to=None,
                                sample_total_codecs=None,
                                full_total_codecs=None,
                                license_link='https://github.com/switchablenorms/DeepFashion2',
                                license_requirements='https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_\
                                h9qQ5BoTYbjvf95aXbid0v2Bw/viewform?usp=sf_link',
                                license_citation="@article{DeepFashion2, author = {Yuying Ge and Ruimao Zhang and \
                                Lingyun Wu and Xiaogang Wang and Xiaoou Tang and Ping Luo}, title={A Versatile \
                                Benchmark for Detection, Pose Estimation, Segmentation and Re-Identification of \
                                Clothing Images}, journal={CVPR}, year={2019}}"
                                # noqa
                                )
        self.save_dataset_metadata(dir_name=f"{self._path_name}", metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_paths_train,
                                       orig_test_paths=orig_paths_test,
                                       classes=self.classes,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc,
                                       many_to_one=True)

        # Delete original directory
        shutil.rmtree(self.path.joinpath('image'))
        shutil.rmtree(self.path.joinpath('annos'))

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
