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
import numpy as np
import random
import shutil
import zipfile
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
from lwll_dataset_prep.logger import log

@dataclass
class deep_fashion (BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'deep_fashion'
    _task_type: str = 'object_detection'
    _fileid: str = '1q9eHZzbTA-USeE_vfZqL5Vn0aTQ4JfHa'
    _fname: str = 'deep-fashion.zip'
    _sample_size_train: int = 200
    _sample_size_test: int = 100

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

    def _get_string(self, bbox: List[int]) -> str:
        bbox_string = ""
        for i, c in enumerate(bbox):
            if (i > 0):
                bbox_string += ", "
            bbox_string += str(c)
        return bbox_string

    def _remove_not_enough_examples(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        # Removes from dataset images with less then 'n' examples
        n = 47
        cl_idx = list(self.classes.keys())
        for cl in cl_idx:
            indexes = df[df['class'] == self.classes[cl]].index
            if (len(indexes) < n):
                df.drop(indexes, inplace=True)
                del self.classes[cl]
        return df

    def _unzip_file(self, filepath: Path, keep_file: bool = False) -> None:
        log.info(f"Unzipping  {filepath}")
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(self.path)

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

        if keep_file is False:
            os.remove(filepath)

    def _correct_keys(self, data: Dict[Any, Any]) -> Dict[Any, Any]:
        # Some paths are missing ".jpg"
        data_new = data.copy()
        for path in data.keys():
            if (path.split('.')[-1] != 'jpg'):
                item = data_new[path]
                del data_new[path]
                path = path.split('.')[0] + '.jpg'
                data_new[path] = item
        return data_new

    def download(self) -> None:
        self.download_google_drive_url(id=self._fileid,
                                       dest=self.path.joinpath(self._fname))
        self._unzip_file(self.path.joinpath(self._fname))
        log.info("Done")

    def process(self) -> None:
        log.info(f"Processing dataset {self._path_name}")

        data: List[Dict[str, Any]] = []
        category_data = np.loadtxt(self.path.joinpath('Anno_coarse/list_category_cloth.txt'), skiprows=2, dtype=str)
        img_category = np.loadtxt(self.path.joinpath('Anno_coarse/list_category_img.txt'), skiprows=2, dtype=str)
        img_bbox = np.loadtxt(self.path.joinpath('Anno_coarse/list_bbox.txt'), skiprows=2, dtype=str)

        # category_data labels are 1-indexed
        self.classes = {i+1: d[0].lower() for i, d in enumerate(category_data)}
        img_to_lb = {img[0]: int(img[1]) for img in img_category}  # indexed by image path
        img_to_bb = {img[0]: [img[1], img[2], img[3], img[4]] for img in img_bbox}  # indexed by image path
        img_to_id = {img: i for i, img in enumerate(img_to_bb.keys())}  # indexed by image path; assign an id to each image path

        # some paths are missing extensions
        img_to_lb = self._correct_keys(img_to_lb)
        img_to_bb = self._correct_keys(img_to_bb)
        img_to_id = self._correct_keys(img_to_id)

        # all images accounted for on both txt files
        assert (img_to_lb.keys() == img_to_bb.keys())

        # The assertions below are True, so there's only one bounding box for each image
        # assert (len(np.unique(img_to_lb.keys())[0]) == len(img_category))
        # assert (len(np.unique(img_to_lb.keys())[0]) == len(img_bbox))

        img_paths = img_to_bb.keys()
        for im_pt in img_paths:
            data.append({'id': f"img_{img_to_id[im_pt]}.jpg",
                         'bbox': self._get_string(img_to_bb[im_pt]),
                         'class': self.classes[img_to_lb[im_pt]],
                         'orig_path': im_pt})

        # Creating dataframe from data
        df = pd.DataFrame(columns=['id', 'label', 'class', 'orig_path']).from_dict(data)
        df = self._remove_not_enough_examples(df)

        all_img_paths = list(img_to_lb.keys()).copy()
        random.Random(5).shuffle(all_img_paths)

        # Split train/test
        train_imgs = all_img_paths[:int(0.7*len(all_img_paths))]
        test_imgs = all_img_paths[int(0.7*len(all_img_paths)):]

        # Create training and test dataframes
        df_train = df[df['orig_path'].isin(train_imgs)]
        df_test = df[df['orig_path'].isin(test_imgs)]

        # Checking leakage
        overlap = list(set(df_train['id']) & set(df_test['id']))
        assert (len(overlap) == 0)

        # Convert to Path
        orig_paths_train = df_train['orig_path'].tolist()
        orig_paths_test = df_test['orig_path'].tolist()
        orig_paths_train = [self.path.joinpath(p) for p in orig_paths_train]
        orig_paths_test = [self.path.joinpath(p) for p in orig_paths_test]

        # new names
        newnames_train = df_train['id'].tolist()
        newnames_test = df_test['id'].tolist()

        df_train = df_train.drop(['orig_path'], axis=1)
        df_test = df_test.drop(['orig_path'], axis=1)

        # Create sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=f"{self._path_name}",
                                                                  df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test,
                                                                  many_to_one=True)
        assert (len(df_train_sample['class'].unique()) == len(self.classes))

        # Checking leakage
        overlap = list(set(df_train_sample['id']) & set(df_test_sample['id']))
        assert (len(overlap) == 0)

        # Move the raw files
        assert(len(orig_paths_train) == len(newnames_train))
        assert(len(orig_paths_test) == len(newnames_test))

        self.original_paths_to_destination(dir_name=f"{self._path_name}",
                                           orig_paths=orig_paths_train,
                                           dest_type='train',
                                           delete_original=False,
                                           new_names=newnames_train)
        self.original_paths_to_destination(dir_name=f"{self._path_name}",
                                           orig_paths=orig_paths_test,
                                           dest_type='test',
                                           delete_original=False,
                                           new_names=newnames_test)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=f"{self._path_name}",
                                                  df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=f"{self._path_name}",
                                                        dataset_type='object_detection',
                                                        sample_number_of_samples_train=len(df_train_sample['id'].unique()),
                                                        sample_number_of_samples_test=len(df_test_sample['id'].unique()),
                                                        sample_number_of_classes=len(df_train_sample['class'].unique()),
                                                        full_number_of_samples_train=len(df_train['id'].unique()),
                                                        full_number_of_samples_test=len(df_test['id'].unique()),
                                                        full_number_of_classes=len(self.classes),
                                                        number_of_channels=3,
                                                        classes=list(self.classes.values()),
                                                        language_from=None,
                                                        language_to=None,
                                                        sample_total_codecs=None,
                                                        full_total_codecs=None,
                                                        license_link='http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html',
                                                        license_requirements='http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html',
                                                        license_citation="@inproceedings{liuLQWTcvpr16DeepFashion, \
                                author = {Liu, Ziwei and Luo, Ping and Qiu, Shi and Wang, Xiaogang and Tang, Xiaoou}, \
                                title = {DeepFashion: Powering Robust Clothes Recognition and Retrieval with Rich Annotations},\
                                booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, \
                                month = {June}, \
                                year = {2016}}"
                                                        # noqa
                                                        )
        self.save_dataset_metadata(dir_name=f"{self._path_name}", metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_paths_train,
                                       orig_test_paths=orig_paths_test,
                                       classes=list(self.classes.values()),
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc,
                                       many_to_one=True)

        # Delete original directory
        shutil.rmtree(self.path.joinpath('Anno_coarse'))
        shutil.rmtree(self.path.joinpath('img'))

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
