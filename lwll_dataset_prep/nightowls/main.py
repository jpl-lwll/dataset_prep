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
from typing import List, Dict, Tuple
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log
import os
import json

@dataclass
class nightowls(BaseProcesser):
    """
    Categories: pedestrain: 1, bicycledriver: 2, motorbikedriver: 3, ignore: 4
    Example annotation:
    {"occluded":null,"difficult":null,"bbox":[453,207,30,54],"id":1000007,"category_id":4,"image_id":1000043,
    "pose_id":5,"tracking_id":1000000,"ignore":1,"area":1620,"truncated":false}

    Excluding the `ignore` category results in 50,225 images in training.
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'nightowls'
    _urls: List[str] = field(default_factory=lambda: ['http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.zip',
                                                      'http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip',
                                                      'http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_training.json',
                                                      'http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.json'])
    _task_type: str = 'object_detection'
    # The _sample_size_* paramters are deprecated with the secondary-seed-label automatic deciding the sample images.
    _sample_size_train: int = 1000
    _sample_size_test: int = 100
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png'])

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
            if 'zip' in fname:
                self.extract_zip(dir_name=self._path_name, fname=fname)

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        image_dict: dict = {}
        image_dict_v: dict = {}
        cate_dict: dict = {}
        cate_dict_v: dict = {}

        ann_list, image_dict, cate_dict = self.parse_json(self.path / f'nightowls_training.json')
        ann_list_v, image_dict_v, cate_dict_v = self.parse_json(self.path / f'nightowls_validation.json')

        classes = set(list(cate_dict.values())[:3])
        print(classes)

        # initializing place holders
        training_ids = []
        training_boxes = []
        training_cates = []
        testing_ids = []
        testing_boxes = []
        testing_cates = []

        # iterate every line in nightowls_training.json
        # bbox is defined as [x_min, y_min, width, height]
        # https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
        for a in ann_list:
            bbox = a["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            cate = a["category_id"]
            img = image_dict[a["image_id"]]
            training_ids.append(img)
            training_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
            training_cates.append(cate_dict[a["category_id"]])

        # process validation subset
        for a in ann_list_v:
            bbox = a["bbox"]
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[0] + bbox[2]
            y_max = bbox[1] + bbox[3]
            cate = a["category_id"]
            img = image_dict_v[a["image_id"]]
            testing_ids.append(img)
            testing_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
            testing_cates.append(cate_dict[cate])

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': training_cates})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': testing_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)

        # Move the raw data files
        orig_train_paths = [self.path.joinpath(f"nightowls_training/{_id}") for _id in df_train['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           delete_original=False, new_names=df_train['id'].unique().tolist())

        orig_test_paths = [self.path.joinpath(f"nightowls_validation/{_id}") for _id in df_test['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test',
                                           delete_original=False, new_names=df_test['id'].unique().tolist())

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='object_detection',
                                 sample_number_of_samples_train=len(df_train_sample["id"].unique()),
                                 sample_number_of_samples_test=len(df_test_sample["id"].unique()),
                                 sample_number_of_classes=len(list(classes)),
                                 full_number_of_samples_train=len(df_train["id"].unique()),
                                 full_number_of_samples_test=len(df_test["id"].unique()),
                                 full_number_of_classes=len(list(classes)),
                                 number_of_channels=3,
                                 classes=list(classes),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://www.nightowls-dataset.org/',
                                 license_requirements='That you include a reference to the Nightowls Dataset in any work that makes use of the dataset.\
                                                       That you do not distribute this dataset or modified versions. It is permissible to distribute\
                                                       derivative works in as far as they are abstract representations of this dataset (such as models\
                                                       trained on it or additional annotations that do not directly include any of our data) and do not\
                                                       allow to recover the dataset or something similar in character. You may not use the dataset or\
                                                       any derivative work for commercial purposes such as, for example, licensing or selling the data,\
                                                       or using the data with a purpose to procure a commercial gain. That all rights not expressly\
                                                      granted to you are reserved by us (University of Oxford).',
                                 license_citation='@inproceedings{Nightowls,title={NightOwls: A pedestrians at night dataset},\
                                                   author={Neumann, Lukas} and Karg, Michelle and Zhang, Shanshan and Scharfenberger,\
                                                   Christian and Piegert, Eric and Mistr, Sarah and Prokofyeva, Olga and Thiel, Robert\
                                                   and Vedaldi, Andrea and Zisserman, Andrew and Schiele, Bernt}, booktitle={Asian\
                                                   Conference on Computer Vision}, pages={691--705}, year={2018}, organization={Springer}}',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validate sample subsets
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
        shutil.rmtree(self.path.joinpath('nightowls_training'))
        shutil.rmtree(self.path.joinpath('nightowls_validation'))
        for p in self._fnames:
            os.remove(self.path.joinpath(p))

        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")

    def parse_json(self, p) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
        # build the image_id -> file_name dict
        image_dict: dict = {}
        cate_dict: dict = {}

        with open(p, 'r') as file:
            jfile = json.load(file)

        records = jfile["images"]
        for r in records:
            _d = {r["id"]: r["file_name"]}
            image_dict = {**image_dict, **_d}

        # build id -> dict
        cates = jfile["categories"]
        for c in cates:
            _d = {c["id"]: c["name"]}
            cate_dict = {**cate_dict, **_d}

        annotations = filter(lambda x: int(x["category_id"]) != 4, jfile["annotations"])
        return annotations, image_dict, cate_dict
