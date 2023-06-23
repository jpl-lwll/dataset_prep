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
import shutil
import pydicom
import cv2
import zipfile
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List
from dataclasses import dataclass, field
from pathlib import Path
from lwll_dataset_prep.logger import log


@dataclass
class chestXray(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'chestXray'
    _task_type: str = 'image_classification'
    _url: str = 'raw/chestXray.zip'
    _fname: str = 'chestXray.zip'
    _sample_size_train: int = 2000
    _sample_size_test: int = 300
    _k_seed: List[int] = field(default_factory=lambda: [1])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

    def get_pulmo_labels(self, label_dir: str, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        for img in df['id']:
            filename = img.split('.')[0]
            lb = filename.split('_')[-1]

            if (lb == "0"):
                df.loc[df['id'] == img, 'class'] = 'normal'
            elif (lb == "1"):
                df.loc[df['id'] == img, 'class'] = 'tuberculosis'
            else:
                print('Error: Invalid filename')
                exit()
        return df

    def _unzip_file(self, filepath: Path) -> None:
        log.info(f"Unzipping  {filepath}")
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(self.data_path)

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

    def download(self) -> None:
        log.info(f"Downloading dataset {self._path_name}")
        # self.download_s3_url(data_path=self._url, dest=self.data_path.joinpath(self._fname))
        self._unzip_file(self.data_path.joinpath(self._fname))
        log.info("Done")

    def process(self) -> None:
        log.info(f"Processing dataset {self._path_name}")
        # Collecting data from pulmonary chest xray - ChinaSet
        df_china = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        chinaset_dir = os.path.join(self.data_path, self._path_name, 'pulmonary chest xray', 'ChinaSet_AllFiles', 'ChinaSet_AllFiles')
        img_list_raw = os.listdir(os.path.join(chinaset_dir, 'CXR_png'))
        df_china['id'] = [img for img in img_list_raw if img.split('.')[-1] == 'png']
        df_china['orig_path'] = [os.path.join(chinaset_dir, 'CXR_png', img) for img in img_list_raw if img.split('.')[-1] == 'png']
        label_dir = os.path.join(chinaset_dir, 'ClinicalReadings')
        df_china = self.get_pulmo_labels(label_dir, df_china)

        # Collecting data from pulmonary chest xray - Montgomery
        df_mont = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        montgomery_dir = os.path.join(self.data_path, self._path_name, 'pulmonary chest xray', 'Montgomery', 'MontgomerySet')
        img_list_raw = os.listdir(os.path.join(montgomery_dir, 'CXR_png'))
        df_mont['id'] = [img for img in img_list_raw if img.split('.')[-1] == 'png']
        df_mont['orig_path'] = [os.path.join(montgomery_dir, 'CXR_png', img) for img in img_list_raw if img.split('.')[-1] == 'png']
        label_dir = os.path.join(montgomery_dir, 'ClinicalReadings')
        df_mont = self.get_pulmo_labels(label_dir, df_mont)

        # Collecting data from rsna dataset
        df_rsna = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        rsna_dir = os.path.join(self.data_path, self._path_name, 'rsna')
        dcm_dir = os.path.join(rsna_dir, 'stage_2_train_images')
        png_dir = os.path.join(rsna_dir, 'train_images_png')
        img_list_dcm = os.listdir(dcm_dir)

        if not os.path.exists(png_dir):
            os.makedirs(png_dir)

        # converting dcm to png
        log.info(f"Converting DCM images to PNG")
        img_list_png = []
        for f in img_list_dcm:
            ds = pydicom.read_file(os.path.join(dcm_dir, f))  # read dicom image
            img = ds.pixel_array  # get image array
            img_path = os.path.join(png_dir, f.replace('.dcm', '.png'))
            cv2.imwrite(img_path, img)  # write png image
            img_list_png.append(img_path)

        df_rsna['id'] = [os.path.basename(p) for p in img_list_png]
        df_rsna['orig_path'] = img_list_png

        classes = pd.read_csv(os.path.join(rsna_dir, 'stage_2_detailed_class_info.csv'))
        pIDs = classes['patientId'].tolist()
        labels = classes['class'].tolist()
        class_dict = dict(zip(pIDs, labels))
        df_rsna['class'] = [class_dict[os.path.basename(i).split('.')[0]] for i in img_list_png]
        df_rsna.loc[df_rsna['class'] == "No Lung Opacity / Not Normal", 'class'] = "no_lung_opacity_not_normal"
        df_rsna.loc[df_rsna['class'] == "Lung Opacity", 'class'] = "lung_opacity"
        df_rsna.loc[df_rsna['class'] == "Normal", 'class'] = "normal"

        dfs = [df_china, df_mont, df_rsna]
        df_data = pd.concat(dfs)
        assert (df_china.shape[0] + df_mont.shape[0] + df_rsna.shape[0] == df_data.shape[0])

        df_data = df_data.sample(frac=1, random_state=2).reset_index(drop=True)
        n = df_data.shape[0]
        df_train = df_data.iloc[:int(0.7 * n), :]
        df_test = df_data.iloc[int(0.7 * n):, :]
        overlap = list(set(df_train['id']) & set(df_test['id']))
        assert (len(overlap) == 0)

        # Extract original paths
        orig_paths_train = df_train['orig_path'].tolist()
        orig_paths_test = df_test['orig_path'].tolist()
        orig_paths_train = [Path(x) for x in orig_paths_train]
        orig_paths_test = [Path(x) for x in orig_paths_test]

        df_train = df_train.drop(['orig_path'], axis=1)
        df_test = df_test.drop(['orig_path'], axis=1)

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name,
                                                                  df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=self._sample_size_train,
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
                                full_number_of_classes=len(df_data['class'].unique()),
                                number_of_channels=3,
                                classes=list(df_data['class'].unique()),
                                language_from=None,
                                language_to=None,
                                sample_total_codecs=None,
                                full_total_codecs=None,
                                license_link="https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities, \
                                              https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data",
                                license_requirements='',
                                license_citation='None')  # noqa
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_paths_train,
                                       orig_test_paths=orig_paths_test,
                                       classes=list(df_data['class'].unique()),
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Removing imgs directory
        shutil.rmtree(self.path.joinpath('pulmonary chest xray'))
        shutil.rmtree(self.path.joinpath('rsna'))
        os.remove(self.data_path.joinpath('chestXray.zip'))
        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
