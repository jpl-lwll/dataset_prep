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
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log
import os
from pycocotools.coco import COCO

@dataclass
class coco2014(BaseProcesser):
    """
    training images: 82783
    val images: 40504

    To parse the bounding boxes from instances_train2014.json and instances_val2014.json, we have to
    use the COCO data PythonAPI https://github.com/cocodataset/cocoapi.
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'coco2014'
    _urls: List[str] = field(default_factory=lambda: ['http://images.cocodataset.org/zips/train2014.zip',
                                                      'http://images.cocodataset.org/zips/val2014.zip',
                                                      'http://images.cocodataset.org/annotations/annotations_trainval2014.zip'])
    _task_type = 'object_detection'
    _sample_size_train: int = 1000
    _sample_size_test: int = 100
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

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        # IMPORTANT NOTE THAT'S NOT MENTIONED IN THIS DATASET README
        # THERE ARE 28204 UNIQUE PATHS AND 3539 UNIQUE NAMES HERE, THIS IS MORE THAN DATA THAT IS
        # ANNOTATED, FOR THIS REASON, WE WILL SELECT A SUBSET OF THE IMAGES WITH ANNOATIONS FURTHER BELOW...

        label_files = [_p for _p in self.path.joinpath('annotations').iterdir() if 'instances' in str(_p)]
        print(label_files)

        training_ids = []
        training_boxes = []
        training_cates = []
        testing_ids = []
        testing_boxes = []
        testing_cates = []

        for file in sorted(label_files):
            coco = COCO(file)
            class_ids = sorted(coco.getCatIds())

            if 'train' in str(file):
                _training_ids = list(coco.imgs.keys())
                print(len(class_ids))
                print(len(_training_ids))

                for training_id in _training_ids:
                    training_anns = coco.loadAnns(coco.getAnnIds(
                        imgIds=[training_id], catIds=class_ids, iscrowd=None
                    ))

                    for ann in training_anns:
                        # ann looks like
                        # {'segmentation': [[233.29, 461.42, 235.35, 395.35, 240.52,
                        # 378.84, 245.68, 372.65, 250.84, 372.65, 283.87, 372.65, 294.19,
                        # ... 362.32, 298.32, 345.81, 301.42, 329.29, 303.48, 308.65,
                        # 238.45, 473.81, 237.42, 473.81]], 'area': 21635.5728,
                        # 'iscrowd': 0, 'image_id': 57870, 'bbox': [233.29, 270.45,
                        # 170.32, 203.36], 'category_id': 62, 'id': 102924}
                        #
                        # the bounding box in COCO comes in {x, y, width, height}
                        # our schema requires {x_min, y_min, x_max, y_max}

                        x, y, width, height = ann['bbox']
                        bbox = f"{int(float(x))}, {int(float(y))}, {int(float(x+width))}, {int(float(y+height))}"
                        cate = ann['category_id']

                        training_ids.append(coco.imgs[training_id]['file_name'])
                        training_boxes.append(bbox)
                        training_cates.append(cate)

            elif 'val' in str(file):
                _testing_ids = list(coco.imgs.keys())
                print(len(class_ids))
                print(len(_testing_ids))

                for testing_id in _testing_ids:
                    testing_anns = coco.loadAnns(coco.getAnnIds(
                        imgIds=[testing_id], catIds=class_ids, iscrowd=None
                    ))
                    for ann in testing_anns:
                        x, y, width, height = ann['bbox']
                        bbox = f"{int(float(x))}, {int(float(y))}, {int(float(x+width))}, {int(float(y+height))}"
                        cate = ann['category_id']
                        testing_ids.append(coco.imgs[testing_id]['file_name'])
                        testing_boxes.append(bbox)
                        testing_cates.append(cate)

        print(len(training_ids), len(training_boxes), len(training_cates), len(testing_ids), len(testing_boxes), len(testing_cates))

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': training_cates})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': testing_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)

        # Move the raw data files
        # print(df_traindf_train['id'].unique().tolist())
        # print(sorted(training_ids)[-5:])
        # print(len(list(set(training_ids))))
        # print(set(training_ids) - set(df_train['id'].unique().tolist()))

        orig_train_paths = [self.path.joinpath('train2014').joinpath(_id) for _id in df_train['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           delete_original=False, new_names=df_train['id'].unique().tolist())

        orig_test_paths = [self.path.joinpath('val2014').joinpath(_id) for _id in df_test['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test',
                                           delete_original=False, new_names=df_test['id'].unique().tolist())

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='object_detection',
                                 sample_number_of_samples_train=len(df_train_sample["id"].unique()),
                                 sample_number_of_samples_test=len(df_test_sample["id"].unique()),
                                 sample_number_of_classes=80,
                                 full_number_of_samples_train=len(df_train["id"].unique()),
                                 full_number_of_samples_test=len(df_test["id"].unique()),
                                 full_number_of_classes=80,
                                 number_of_channels=3,
                                 classes=list(set(training_cates)),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='http://cocodataset.org/#home',
                                 license_requirements='None',
                                 license_citation='http://cocodataset.org/#termsofuse',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=list(set(training_cates)), train_df=df_train,
                                       test_df=df_test, sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample, metadata=dataset_doc,
                                       many_to_one=True)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        shutil.rmtree(self.path.joinpath('train2014'))
        os.remove(self.path.joinpath('train2014.zip'))
        shutil.rmtree(self.path.joinpath('val2014'))
        os.remove(self.path.joinpath('val2014.zip'))
        shutil.rmtree(self.path.joinpath('annotations'))
        os.remove(self.path.joinpath('annotations_trainval2014.zip'))
        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")
