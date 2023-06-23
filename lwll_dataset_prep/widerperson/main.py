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
class widerperson(BaseProcesser):
    '''
    8000/1000 images in training and validation

    ########## Annotation Format ##########
    Each image of training and valiadation subsets in the "./Images" folder
    (e.g., 000001.jpg) has a corresponding annotation text file in the "./Annotations"
    folder (e.g., 000001.jpg.txt). The annotation file structure is in the following format:

    < number of annotations in this image = N >
    < anno 1 >
    < anno 2 >
    ......
    < anno N >

    where one object instance per row is [class_label, x1, y1, x2, y2], and the class label definition is:

    class_label =1: pedestrians
    class_label =2: riders
    class_label =3: partially-visible persons
    class_label =4: ignore regions
    class_label =5: crowd
    '''
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'widerperson'
    _task_type: str = 'object_detection'
    _fnames: List[str] = field(default_factory=lambda: ['WiderPerson.zip'])
    _urls: List[str] = field(default_factory=lambda: ['1I7OjhaomWqd8Quf7o5suwLloRlY0THbp'])
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

        label_csv_train = self.path.joinpath('train.txt')
        label_csv_val = self.path.joinpath('val.txt')

        with open(label_csv_train) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                anno_path = self.path.joinpath(f"Annotations/{row[0]}.jpg.txt")
                with open(anno_path) as anno_file:
                    lines = anno_file.readlines()
                    for line in lines[1:]:
                        words = line.split(' ')
                        cate = words[0]
                        if cate == "3" or cate == "4":
                            continue
                        x_min = int(words[1])
                        y_min = int(words[2])
                        x_max = int(words[3])
                        y_max = int(words[4])
                        training_ids.append(f"{row[0]}.jpg")
                        training_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
                        training_cates.append(cate)
                        orig_train_paths.append(self.path.joinpath(f"Images/{row[0]}.jpg"))

        with open(label_csv_val) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                anno_path = self.path.joinpath(f"Annotations/{row[0]}.jpg.txt")
                with open(anno_path) as anno_file:
                    lines = anno_file.readlines()
                    for line in lines[1:]:
                        words = line.split(' ')
                        cate = words[0]
                        if cate == "3" or cate == "4":
                            continue
                        x_min = int(words[1])
                        y_min = int(words[2])
                        x_max = int(words[3])
                        y_max = int(words[4])
                        testing_ids.append(f"{row[0]}.jpg")
                        testing_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
                        testing_cates.append(cate)
                        orig_test_paths.append(self.path.joinpath(f"Images/{row[0]}.jpg"))

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
                                 license_link='http://www.cbsr.ia.ac.cn/users/sfzhang/WiderPerson/',
                                 license_requirements='',
                                 license_citation='@article{zhang2019widerperson, Author = {Zhang, Shifeng\
                                 and Xie, Yiliang and Wan, Jun and Xia, Hansheng and Li, Stan Z. and Guo,\
                                 Guodong}, journal = {IEEE Transactions on Multimedia (TMM)}, Title =\
                                 {WiderPerson: A Diverse Dataset for Dense Pedestrian Detection in the Wild},\
                                 Year = {2019}}',  # noqa
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
        for f in ['ReadMe.txt', 'train.txt', 'val.txt', 'test.txt', 'WiderPerson.zip']:
            os.remove(self.path.joinpath(f))
        for d in ['Annotations', 'Evaluation', 'Images']:
            shutil.rmtree(self.path.joinpath(d))
        log.info("Done")
        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
