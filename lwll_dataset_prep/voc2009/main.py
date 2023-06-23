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
import random
import xml.etree.ElementTree as ET

@dataclass
class voc2009(BaseProcesser):
    """
    Number of images: 7818
    We divide the images to train/val with 80:20 ratio
    The 80:20 ratio is with regard to images, otherwise training and testing will have overlapped images

    20 categories in total:
    Person: person
    Animal: bird, cat, cow, dog, horse, sheep
    Vehicle: aeroplane, bicycle, boat, bus, car, motorbike, train
    Indoor: bottle, chair, dining table, potted plant, sofa, tv/monitor
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'voc2009'
    _annotation_path: Path = data_path.joinpath(_path_name).joinpath('VOCdevkit/VOC2009/Annotations')
    _img_path: Path = data_path.joinpath(_path_name).joinpath('VOCdevkit/VOC2009/JPEGImages')
    _urls: List[str] = field(default_factory=lambda: ['http://host.robots.ox.ac.uk/pascal/VOC/voc2009/VOCtrainval_11-May-2009.tar'])
    _task_type: str = 'object_detection'
    # The _sample_size_* paramters are deprecated with the secondary-seed-label automatic deciding the sample images.
    _sample_size_train: int = 1000
    _sample_size_test: int = 100
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def _find(self, element: ET.Element, tag: str) -> ET.Element:
        result = element.find(tag)
        assert result is not None, ('No tag "{tag}" found '
                                    'in element "{element}".'
                                    .format(tag=tag,
                                            element=element))
        return result

    def _text(self, element: ET.Element) -> str:
        result = element.text
        assert result is not None, ('No tag "text" found '
                                    'in element "{element}".'
                                    .format(element=element))
        return result

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        label_files = [_p for _p in self._annotation_path.iterdir()]
        # shuffle the label_files
        random.Random(4).shuffle(label_files)

        classes = set()

        split_idx = int(len(label_files) * 0.8)
        paths_train = label_files[:split_idx]
        paths_test = label_files[split_idx:]

        training_ids = []
        training_boxes = []
        training_cates = []
        testing_ids = []
        testing_boxes = []
        testing_cates = []

        for p in paths_train:
            tree = ET.parse(str(p))
            root = tree.getroot()
            '''
            <annotation>
            <filename>2009_003722.jpg</filename>
            <folder>VOC2009</folder>
            <object>
                <name>person</name>
                <bndbox>
                    <xmax>376</xmax>
                    <xmin>215</xmin>
                    <ymax>363</ymax>
                    <ymin>46</ymin>
                </bndbox>
                <difficult>0</difficult>
                <occluded>1</occluded>
                <pose>Left</pose>
                <truncated>0</truncated>
            </object>
            <object>
                <name>motorbike</name>
                <bndbox>
                    <xmax>458</xmax>
                    <xmin>33</xmin>
                    <ymax>340</ymax>
                    <ymin>71</ymin>
                </bndbox>
                <difficult>0</difficult>
                <occluded>1</occluded>
                <pose>Left</pose>
                <truncated>0</truncated>
            </object>
            <object>
                <name>chair</name>
                <bndbox>
                    <xmax>303</xmax>
                    <xmin>280</xmin>
                    <ymax>70</ymax>
                    <ymin>41</ymin>
                </bndbox>
                <difficult>0</difficult>
                <occluded>0</occluded>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
            </object>
            <object>
                <name>car</name>
                <bndbox>
                    <xmax>292</xmax>
                    <xmin>1</xmin>
                    <ymax>181</ymax>
                    <ymin>1</ymin>
                </bndbox>
                <difficult>0</difficult>
                <occluded>1</occluded>
                <pose>Unspecified</pose>
                <truncated>1</truncated>
            </object>
            <segmented>0</segmented>
            <size>
                <depth>3</depth>
                <height>375</height>
                <width>500</width>
            </size>
            <source>
                <annotation>PASCAL VOC2009</annotation>
                <database>The VOC2009 Database</database>
                <image>flickr</image>
            </source>
            </annotation>
            '''

            fname_elt = self._find(root, 'filename')
            fname = self._text(fname_elt)

            for obj in root.findall('object'):
                cls_elt = self._find(obj, 'name')
                cls = self._text(cls_elt)
                bbox = self._find(obj, 'bndbox')

                x_max_elt = self._find(bbox, 'xmax')
                x_min_elt = self._find(bbox, 'xmin')
                y_max_elt = self._find(bbox, 'ymax')
                y_min_elt = self._find(bbox, 'ymin')

                x_max = self._text(x_max_elt)
                x_min = self._text(x_min_elt)
                y_max = self._text(y_max_elt)
                y_min = self._text(y_min_elt)
                classes.add(cls)
                training_cates.append(cls)
                training_boxes.append(f"{int(float(x_min))}, {int(float(y_min))}, {int(float(x_max))}, {int(float(y_max))}")
                training_ids.append(fname)

        for p in paths_test:
            tree = ET.parse(str(p))
            root = tree.getroot()

            fname_elt = self._find(root, 'filename')
            fname = self._text(fname_elt)

            for obj in root.findall('object'):
                cls_elt = self._find(obj, 'name')
                cls = self._text(cls_elt)
                bbox = self._find(obj, 'bndbox')

                x_max_elt = self._find(bbox, 'xmax')
                x_min_elt = self._find(bbox, 'xmin')
                y_max_elt = self._find(bbox, 'ymax')
                y_min_elt = self._find(bbox, 'ymin')

                x_max = self._text(x_max_elt)
                x_min = self._text(x_min_elt)
                y_max = self._text(y_max_elt)
                y_min = self._text(y_min_elt)

                testing_cates.append(cls)
                testing_boxes.append(f"{x_min}, {y_min}, {x_max}, {y_max}")
                testing_ids.append(fname)

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'bbox': training_boxes, 'class': training_cates})
        df_test = pd.DataFrame({'id': testing_ids, 'bbox': testing_boxes, 'class': testing_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=True)

        # Move the raw data files
        print(self._img_path)

        orig_train_paths = [self._img_path.joinpath(_id) for _id in df_train['id'].unique().tolist()]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           delete_original=False, new_names=df_train['id'].unique().tolist())

        orig_test_paths = [self._img_path.joinpath(_id) for _id in df_test['id'].unique().tolist()]
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
                                 license_link='http://host.robots.ox.ac.uk/pascal/VOC/voc2009/index.html',
                                 license_requirements='None',
                                 license_citation='@misc{pascal-voc-2009, author = "Everingham, M. and Van~Gool, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.", title = "The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2009 {(VOC2009)} {R}esults", showpublished = "http://www.pascal-network.org/challenges/VOC/voc2009/workshop/index.html"}',  # noqa
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
        shutil.rmtree(self.path.joinpath('VOCdevkit/'))
        os.remove(self.path.joinpath('VOCtrainval_11-May-2009.tar'))

        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")
