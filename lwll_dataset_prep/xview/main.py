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
from typing import List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import shutil
from lwll_dataset_prep.logger import log
import os
import cv2
import glob
import json
import yaml


@dataclass
class xview(BaseProcesser):
    """
    Number of images: 1413

    classes: 60 (e.g. aircraft hangar, passenger car, shipping container)
    training images: 847 (600345 objects)
    validation images: 282 (200291 objects)
    testing images: 284

    WARNING: The sample dataset does not guarantee to cover all 60 classes

    The source data is organized as follows:

        xview
        ├── train_images_small
        │   ├── 1.tif
        │   ├── 2.tif
        │   .
        │   .
        │   .
        │   └── N.tif
        ├── val_images_small
        │   ├── 100.tif
        │   ├── 101.tif
        │   └── 102.tif
        │   .
        │   .
        │   .
        │   └── M.tif
        └── xView_train.geojson
    """

    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'xview'

    _urls: List[str] = field(default_factory=lambda:
                             ['https://data.kitware.com/api/v1/item/5f36ecb39014a6d84e7757ef/download',
                              'https://data.kitware.com/api/v1/item/5f36dfc49014a6d84e7739e1/download'])

    _task_type: str = 'object_detection'

    # Numbers are a guess
    _sample_size_train: int = 100
    _sample_size_test: int = 60
    _k_seed: List[int] = field(default_factory=lambda: [1])

    _valid_extensions: List[str] = field(default_factory=lambda: ['.tif'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)

        self._fnames: List[str] = [
            'train_images.tgz',
            'train_labels.tgz',
        ]

        self.full_path = str(self.data_path.joinpath(
            self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(
            self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def is_faulty(self, img: np.ndarray, thrs: int = 20) -> bool:
        # checks if image has 'thrs' black columns
        count = 0
        for x in range(img.shape[1]):
            sum_cols = 0
            for y in range(img.shape[0]):
                sum_cols += img[y][x]
            if sum_cols == 0:
                count += 1

            if count > thrs:
                return True
        return False

    def remove_faulty_imgs(self) -> List[str]:
        removed = []
        img_path = os.path.join(self.path, 'train_images')
        img_list = os.listdir(img_path)
        for im in img_list:
            img_gray = cv2.cvtColor(cv2.imread(os.path.join(img_path, im)), cv2.COLOR_BGR2GRAY)
            faulty = self.is_faulty(img_gray)
            if (faulty):
                os.remove(os.path.join(img_path, im))
                removed.append(im)
                log.info(f'Faulty image removed: {im}')
                # shutil.move(os.path.join(img_path, im), os.path.join(remove_path, im))
        return removed

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=True)
            log.info("Done")

    def process(self) -> None:
        log.info('Processing xview dataset')

        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        cur_dir_path = Path(os.path.dirname(os.path.realpath(__file__)))
        with open(cur_dir_path / 'class_mappings.yaml', 'r') as file:
            self._class_ids_map: Dict[int, str] = yaml.load(file)

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        # Removing annoying hidden files
        for fname in glob.glob(f'{self.path}/train_images/.*'):
            os.remove(fname)

        log.info('Removing faulty images from dataset')
        removed = self.remove_faulty_imgs()
        # removed = ['2004.tif', '1450.tif', '1863.tif', '1081.tif' , '2012.tif' , '100.tif', '2599.tif', '2000.tif','1870.tif', \
        # '1988.tif', '2306.tif', '1237.tif', '1154.tif', '89.tif', '2314.tif', '2513.tif', '84.tif', '90.tif', '1429.tif', '2530.tif', \
        # '1372.tif', '95.tif', '1164.tif', '2309.tif', '83.tif', '752.tif', '1459.tif', '1465.tif', '481.tif', '2542.tif', '1739.tif', \
        # '1880.tif', '1072.tif', '2591.tif', '1891.tif']

        # get labels
        log.info('Getting labels')
        with open(self.path / f'xView_train.geojson', 'r') as file:
            labels = json.load(file)['features']

        # convert ann to dict
        anns = list()
        for lab in labels:
            class_label = lab['properties']['type_id']
            id_label = lab['properties']['image_id']
            bbox_label = lab['properties']['bounds_imcoords']
            # Skipping some class labels https://github.com/DIUx-xView/xView1_baseline/issues/3
            if class_label in [75, 82] or id_label == '1395.tif':
                continue
            anns.append(dict({
                'id': id_label,
                'bbox': bbox_label,
                'class': self._map_class_id_to_name(class_label)
            }))

        # convert to pandas
        anns_df: pd.DataFrame = pd.DataFrame(anns)
        anns_df = anns_df.sample(frac=1, random_state=34).reset_index(drop=True)

        # Get Classes
        classes = anns_df['class'].unique().tolist()
        image_fns = list(anns_df['id'].unique().tolist())
        # Split dataset by image name
        # TODO: Use interger programming method:
        #  https://github.com/bmreiniger/datascience.stackexchange/blob/master/54450.ipynb
        n_train = int(len(image_fns) * 0.7)
        train_fns = image_fns[:n_train]
        test_fns = image_fns[n_train:]
        # Manually pull out 107.tif from train to test to enforce 60 classes in test
        train_fns.remove('107.tif')
        test_fns.append('107.tif')

        df_train = anns_df[anns_df['id'].isin(train_fns)]
        df_test = anns_df[anns_df['id'].isin(test_fns)]

        # removing filtered images
        df_train = anns_df[~anns_df['id'].isin(removed)]
        df_test = anns_df[~anns_df['id'].isin(removed)]

        log.info(f'Train has {len(df_train["class"].unique())} classes and '
                 f'Test has {len(df_test["class"].unique())} classes')

        train_full_fns = self.get_full_fname_unqiue_paths(df_train)
        test_full_fns = self.get_full_fname_unqiue_paths(df_test)

        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=train_full_fns,
                                           dest_type='train', delete_original=False)
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=test_full_fns,
                                           dest_type='test', delete_original=False)

        # Create our sample subsets
        # Per performer's request, generating samples exactly the same as train/val data
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name,
                                                                  df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=len(df_train),
                                                                  samples_test=len(df_test),
                                                                  many_to_one=True)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name,
                                                  df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='object_detection',
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
                                 license_link='https://challenge.xviewdataset.org/data-download',
                                 license_requirements='agree_to_termsLicense Link at '
                                                      'https://challenge.xviewdataset.org/data-download',
                                 license_citation='N/A',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=train_full_fns,
                                       orig_test_paths=test_full_fns,
                                       classes=classes,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc,
                                       many_to_one=True)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in self._fnames:
            # We assume the tar files have no '.'s in their name before `.tar.gz` or just `.tar`
            d = f.split('.')[0]
            if d == 'train_labels':
                os.unlink(self.path.joinpath('xView_train.geojson'))
                os.unlink(self.path.joinpath(f))
            else:
                shutil.rmtree(self.path.joinpath(d))
                os.unlink(self.path.joinpath(f))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name,
                                dataset_type='development',
                                task_type=self._task_type)
        log.info("Done")

    def _map_class_id_to_name(self, class_id: int) -> str:
        """
        Args:
            class_id: number for the class

        Return:
            class name
        """
        return self._class_ids_map[class_id]

    def get_full_fname_unqiue_paths(self, anns: pd.DataFrame) -> np.array:
        """ Add full path to the image filenanes in ID.

        """
        return list(map(lambda x: Path().joinpath(f'{self.path}/train_images/', x),
                        sorted(anns['id'].unique().tolist())))
