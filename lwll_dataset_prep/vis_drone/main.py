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
import csv
from typing import List
from pathlib import Path
import pandas as pd
from lwll_dataset_prep.logger import log
import os
import shutil

@dataclass
class vis_drone(BaseProcesser):
    """
    Source data: https://drive.google.com/file/d/1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn/view?usp=sharing
    https://drive.google.com/file/d/1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59/view?usp=sharing

    Annoation Description: http://aiskyeye.com/evaluate/results-format/
    Training images: 6,471, objects: 353,550
    Test images: 548, objects: 40,169
    Annotation:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

    Categories:
    ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7),
    awning-tricycle (8), bus (9), motor (10), others (11)
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'vis_drone'
    _task_type: str = 'object_detection'
    _fnames: List[str] = field(default_factory=lambda: ['VisDrone2019-DET-train.zip',
                                                        'VisDrone2019-DET-val.zip'])
    _urls: List[str] = field(default_factory=lambda: ['1a2oHjcEcwXP8oUF95qiwrqzACb2YlUhn',
                                                      '1bxK5zgLn0_L8x276eKkuYA_FzwCIjb59'])
    _sample_size_train: int = 500
    _sample_size_test: int = 100
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        # We need to enable S3 downloading in self.download_data_from_url
        # For now, the dataset is donwloaded separatedly
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, drive_download=True, overwrite=False)
        log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_zip(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        training_ids = []
        training_boxes = []
        training_cates = []
        testing_ids = []
        testing_boxes = []
        testing_cates = []
        orig_train_paths = []
        orig_test_paths = []

        label_csv_train = self.path.joinpath('VisDrone2019-DET-train/annotations')
        label_csv_val = self.path.joinpath('VisDrone2019-DET-val/annotations')

        for f in os.listdir(label_csv_train):
            fpath = label_csv_train.joinpath(f)
            base_name = f.split('.')[0]
            with open(fpath) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    cate = row[5]
                    if cate == "0" or cate == "11":
                        continue
                    training_ids.append(f"{base_name}.jpg")
                    x_min = int(row[0])
                    y_min = int(row[1])
                    width = int(row[2])
                    height = int(row[3])
                    x_max = x_min + width
                    y_max = y_min + height
                    training_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
                    training_cates.append(cate)
                    orig_train_paths.append(self.path.joinpath(f"VisDrone2019-DET-train/images/{base_name}.jpg"))

        for f in os.listdir(label_csv_val):
            fpath = label_csv_val.joinpath(f)
            base_name = f.split('.')[0]
            with open(fpath) as csv_file:
                reader = csv.reader(csv_file)
                for row in reader:
                    cate = row[5]
                    if cate == "0" or cate == "11":
                        continue
                    testing_ids.append(f"{base_name}.jpg")
                    x_min = int(row[0])
                    y_min = int(row[1])
                    width = int(row[2])
                    height = int(row[3])
                    x_max = x_min + width
                    y_max = y_min + height
                    testing_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
                    testing_cates.append(cate)
                    orig_test_paths.append(self.path.joinpath(f"VisDrone2019-DET-val/images/{base_name}.jpg"))

        classes = list(set(training_cates))

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': training_cates})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': testing_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=list(set(orig_train_paths)), dest_type='train', delete_original=False)

        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=list(set(orig_test_paths)), dest_type='test', delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

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
                                 license_link='http://aiskyeye.com/data-protection/',
                                 license_requirements='The copyright of the VisDrone2020 dataset is reserved\
                                 by the AISKYEYE team at Lab of Machine Learning and Data Mining, Tianjin\
                                 University, China. The dataset described on this page is distributed under\
                                 the  Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License, which\
                                 implies that you must: (1) attribute the work as specified by the original authors;\
                                 (2) may not use this work for commercial purposes ; (3) if you alter, transform,\
                                 or build upon this work, you may distribute the resulting work only under the same\
                                 license. The dataset is provided “as it is” and we are not responsible for any\
                                 subsequence from using this dataset.',
                                 license_citation='@article{zhu2020vision, title={Vision Meets Drones: Past, Present\
                                 and Future},author={Zhu, Pengfei and Wen, Longyin and Du, Dawei and Bian, Xiao and\
                                  Hu, Qinghua and Ling, Haibin},journal={arXiv preprint arXiv:2001.06303},year={2020}}',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=list(set(training_cates)),
                                       train_df=df_train, test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc,
                                       many_to_one=True)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in self._fnames:
            # We assume the tar files have no '.'s in their name before `.tar.gz` or just `.tar`
            d = f.split('.')[0]
            shutil.rmtree(self.path.joinpath(d))
            os.remove(self.path.joinpath(f))
        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
