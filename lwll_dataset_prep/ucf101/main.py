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
import cv2
import requests
import urllib3
import shutil
import glob
import random
import copy
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from lwll_dataset_prep.logger import log
from rarfile import RarFile

@dataclass
class ucf101(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'ucf101'
    _task_type: str = 'video_classification'
    _url: str = 'https://www.crcv.ucf.edu/data/UCF101/UCF101.rar'
    _fname: str = 'UCF101.rar'
    _sample_size_train: int = 1000
    _sample_size_test: int = 100
    _valid_extensions: List[str] = field(default_factory=lambda: ['avi'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(f'{self._path_name}_data')
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample')

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

        self.classes: List[Any] = []
        self.data: List[Dict[Any, Any]] = []
        self.segment_id: int = 0

    def _extract_rar(self) -> None:
        rf = RarFile(self.data_path.joinpath(self._fname))
        rf.extractall(self.data_path)

    def _get_dir_contents(self, pathdir: str) -> List[str]:
        contents = os.listdir(pathdir)
        if '.DS_Store' in contents:
            contents.remove('.DS_Store')
        return contents

    def _get_clip_paths(self, pathdir: Path) -> List[str]:
        allfiles = self._get_dir_contents(str(pathdir))
        return [str(pathdir.joinpath(clip)) for clip in allfiles if clip.split('.')[-1] in self._valid_extensions]

    def _process_video(self, vidpath: str, dstype: str, segid: int, start_frame: int) -> int:
        video = cv2.VideoCapture((str(vidpath)))
        log.info(f"Processing clip {str(vidpath)}")
        end_frame = copy.copy(start_frame)

        frame_dir = self.full_path.joinpath(dstype).joinpath(str(segid))
        if not os.path.isdir(frame_dir):
            os.makedirs(frame_dir)

        while(video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            frame_path = self.full_path.joinpath(dstype).joinpath(str(segid)).joinpath(f"{end_frame}.jpg")
            cv2.imwrite(str(frame_path), frame)
            end_frame += 1

        video.release()
        return end_frame-1

    def _check_labels_traintest(self, vpaths: List[str], train_ids: List[int], test_ids: List[int]) -> bool:
        train_labels = set()
        test_labels = set()

        for i, vid in enumerate(vpaths):
            if i in train_ids:
                train_labels.add(os.path.basename(os.path.dirname(vid)))
            elif i in test_ids:
                test_labels.add(os.path.basename(os.path.dirname(vid)))
            else:
                print('Error: missing id', i)
                exit()
        try:
            assert (len(train_labels & set(self.classes)) == len(self.classes))
            assert (len(test_labels & set(self.classes)) == len(self.classes))
            return True
        except AssertionError:
            return False

    def split_traintest(self, vpaths: List[str]) -> Tuple[List[int], List[int]]:
        n_clips = len(vpaths)
        seed_n = 0
        all_labels_train = False
        while all_labels_train is False:
            all_ids = np.arange(n_clips)
            np.random.seed(seed_n)
            np.random.shuffle(all_ids)
            train_ids = all_ids[:int(0.8*(len(all_ids)))]
            test_ids = all_ids[int(0.8*(len(all_ids))):]
            all_labels_train = self._check_labels_traintest(vpaths, train_ids, test_ids)
            seed_n += 1
        assert (len(set(train_ids) & set(test_ids)) == 0)
        return train_ids, test_ids

    def download(self) -> None:
        log.info(f"Downloading dataset {self._path_name}")
        urllib3.disable_warnings()
        r = requests.get(self._url, verify=False)
        # Download file
        open(self.data_path.joinpath(self._fname), 'wb').write(r.content)
        # Uncompress and rename directory
        log.info(f"Extracting file {self._fname}")
        self._extract_rar()
        os.rename(os.path.join(str(self.data_path), 'UCF-101'), os.path.join(str(self.data_path), f'{self._path_name}_data'))
        log.info(f"Removing file {self._fname}")
        os.remove(self.data_path.joinpath(self._fname))
        return

    def process(self) -> None:
        log.info(f"Processing {self._path_name}...")
        self.classes = self._get_dir_contents(str(self.path))
        self.df_train: List[Dict[Any, Any]] = []
        self.df_test: List[Dict[Any, Any]] = []

        vpaths = glob.glob(str(self.path.joinpath("*/*")))
        random.Random(4).shuffle(vpaths)

        # split train/test clips
        train_ids, test_ids = self.split_traintest(vpaths)
        segment_id = -1
        start_frame = 0

        for vpath in vpaths:
            label = os.path.basename(os.path.dirname(vpath))
            segment_id += 1

            if (segment_id in train_ids):
                end_frame = self._process_video(vpath, 'train', segment_id, start_frame)
                self.df_train.append({'id': str(segment_id),
                                      'video_id': str(segment_id),
                                      'start_frame': start_frame,
                                      'end_frame': end_frame,
                                      'class': str(label)})
            elif (segment_id in test_ids):
                end_frame = self._process_video(vpath, 'test', segment_id, start_frame)
                self.df_test.append({'id': str(segment_id),
                                     'video_id': str(segment_id),
                                     'start_frame': start_frame,
                                     'end_frame': end_frame,
                                     'class': str(label)})
            else:
                print('Error: Invalid segment id')
                exit()

            start_frame = end_frame+1

        assert (segment_id == len(vpaths)-1)

        self.df_train = pd.DataFrame(columns={'id', 'video_id', 'start_frame', 'end_frame', 'class'}).from_dict(self.df_train)
        self.df_test = pd.DataFrame(columns={'id', 'video_id', 'start_frame', 'end_frame', 'class'}).from_dict(self.df_test)

        # Create our sample subsets
        self.df_train_sample, self.df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=self.df_train,
                                                                            df_test=self.df_test, samples_train=self._sample_size_train,
                                                                            samples_test=self._sample_size_test)
        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=self.df_train_sample, df_test_sample=self.df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='video_classification',
                                 sample_number_of_samples_train=len(self.df_train_sample),
                                 sample_number_of_samples_test=len(self.df_test_sample),
                                 sample_number_of_classes=self.df_train_sample['class'].nunique(),
                                 full_number_of_samples_train=len(self.df_train),
                                 full_number_of_samples_test=len(self.df_test),
                                 full_number_of_classes=len(self.classes),
                                 number_of_channels=3,
                                 classes=self.classes,
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://www.crcv.ucf.edu/data/UCF101.php',
                                 license_requirements='None',
                                 license_citation='Khurram Soomro, Amir Roshan Zamir and Mubarak Shah, UCF101: A Dataset of 101 Human Action Classes From Videos in The Wild., CRCV-TR-12-01, November, 2012.')  # noqa
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        # orig_train_paths = os.listdir(self.full_path.joinpath('train'))
        # orig_test_paths = os.listdir(self.full_path.joinpath('test'))
        # log.info("Validating sample datasets")
        # self.validate_output_structure(name=self._path_name,
        #                              orig_train_paths=orig_train_paths,
        #                              orig_test_paths=orig_test_paths,
        #                              classes=self.classes,
        #                              train_df=self.df_train,
        #                              test_df=self.df_test,
        #                              sample_train_df=self.df_train_sample,
        #                              sample_test_df=self.df_test_sample,
        #                              metadata=dataset_doc)

        log.info(f"Removing original data directory {self.path}")
        shutil.rmtree(self.path)

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
