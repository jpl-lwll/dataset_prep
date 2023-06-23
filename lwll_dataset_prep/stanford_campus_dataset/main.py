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
import cv2
import zipfile
import shutil
import pandas as pd
import numpy as np
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from pathlib import Path
from lwll_dataset_prep.logger import log

@dataclass
class stanford_campus_dataset(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'stanford_campus_dataset'
    _task_type: str = 'object_detection'
    _url: str = 'http://vatic2.stanford.edu/stanford_campus_dataset.zip'
    _fname: str = 'stanford_campus_dataset.zip'
    _sample_size_train: int = 600
    _sample_size_test: int = 120

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

        self.data: List[Dict[str, Any]] = []
        self.classes: List[Any] = []
        self.vid_n: int = 0

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

    def _unzip_file(self, filepath: Path, keep_file: bool = False) -> None:
        with zipfile.ZipFile(filepath, "r") as zf:
            zf.extractall(self.path)

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

        if keep_file is False:
            os.remove(filepath)

    def _get_paths(self) -> Tuple[List, List]:

        vid_scene_paths = [os.path.join(self.path, "videos", scene)
                           for scene in os.listdir(os.path.join(self.path, "videos"))
                           if os.path.isdir(os.path.join(self.path, "videos", scene))]
        lb_scene_paths = [os.path.join(self.path, "annotations", scene)
                          for scene in os.listdir(os.path.join(self.path, "annotations"))
                          if os.path.isdir(os.path.join(self.path, "annotations", scene))]

        vid_paths: List[str] = []
        label_paths: List[str] = []
        for vp, lp in zip(vid_scene_paths, lb_scene_paths):
            videos = [os.path.join(vp, name) for name in os.listdir(vp) if os.path.isdir(os.path.join(vp, name))]
            labels = [os.path.join(lp, name) for name in os.listdir(lp) if os.path.isdir(os.path.join(lp, name))]

            vid_paths += videos
            label_paths += labels

        return vid_paths, label_paths

    def _select_frames(self, lpath: str) -> Dict:
        lb_content = pd.read_csv(os.path.join(lpath, 'annotations.txt'), sep=" ")
        lb_content.columns = ['trackID', 'xmin', 'ymin', 'xmax', 'ymax', 'frame', 'lost', 'occluded', 'generated', 'label']
        lb_content = lb_content[lb_content['lost'] == 0]  # drop lost labels
        n_frames = lb_content['frame'].max()
        lb_content = lb_content.sort_values(by='frame')
        select = [n for n in list(np.arange(n_frames)) if n % 50 == 0]  # selects subset of labels
        lb_content = lb_content[lb_content['frame'].isin(select)]

        chosen: Dict[int, List[str]] = dict()  # dict indexed by frame number

        for i, fr in enumerate(lb_content['frame']):
            try:
                chosen[fr].append(lb_content.iloc[i, :])
            except KeyError:
                chosen[fr] = []
                chosen[fr].append(lb_content.iloc[i, :])

        return chosen

    def _fix_df_structure(self, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:

        df_fixed = pd.DataFrame(columns=['id', 'bbox', 'class'])

        for i, row in enumerate(df.iterrows()):
            img_name = row[1]['id']
            boxes = row[1]['bbox']
            classes = row[1]['class']

            assert(len(boxes) == len(classes))

            for box, cl in zip(boxes, classes):
                df_fixed = df_fixed.append({'id': img_name, 'bbox': box[0], 'class': cl[0]}, ignore_index=True)

        return df_fixed

    def _extract_frames(self, vpath: str, lpath: str) -> None:
        log.info(f"Processing video {vpath}")
        chosen = self._select_frames(lpath)
        self.vid_n += 1

        video = cv2.VideoCapture(os.path.join(vpath, 'video.mov'))
        fr_n = 0

        while (video.isOpened()):
            ret, frame = video.read()

            if not ret:
                break

            if (fr_n in chosen.keys()):
                bboxes, lb_all = [], []  # lb_all contain all labels from current frame
                for i in range(len(chosen[fr_n])):  # Loop over all objects in frame
                    x_min, y_min = chosen[fr_n][i]['xmin'], chosen[fr_n][i]['ymin']
                    x_max, y_max = chosen[fr_n][i]['xmax'], chosen[fr_n][i]['ymax']
                    bbox = [f"{x_min}, {y_min}, {x_max}, {y_max}"]
                    lb = chosen[fr_n][i]['label']
                    bboxes.append(bbox)
                    lb_all.append([lb])

                    if (lb not in self.classes):
                        self.classes.append(lb)

                # Add objects from current frame to data
                frame_dir = self.path.joinpath("tmp_imgs")
                frame_path = os.path.join(frame_dir, f"{self.vid_n}_{fr_n}.jpg")
                cv2.imwrite(frame_path, frame)

                self.data.append({'id': f"{self.vid_n}_{fr_n}.jpg",
                                  'bbox': bboxes,
                                  'class': lb_all,
                                  'orig_path': frame_path})
            fr_n += 1

        video.release()

        return

    def download(self) -> None:
        log.info(f"Downloading dataset {self._path_name}")
        self.download_url(url=self._url, dest=self.data_path.joinpath(self._fname))
        self._unzip_file(self.data_path.joinpath(self._fname))
        log.info("Done")

    def process(self) -> None:

        vid_paths, label_paths = self._get_paths()

        # Creates temporary directory for frames extracted from videos
        if not os.path.isdir(os.path.join(self.path, "tmp_imgs")):
            os.makedirs(os.path.join(self.path, "tmp_imgs"))

        for vpath, lpath in zip(vid_paths, label_paths):
            # makes sure we're processing video and respective annotation
            assert(vpath.split('/')[-2:] == lpath.split('/')[-2:])
            self._extract_frames(vpath, lpath)

        # Finalizing dataset
        df = pd.DataFrame(columns=['id', 'label', 'class', 'orig_path']).from_dict(self.data)

        # Shuffle and split into train and test sets
        df_domain = df.sample(frac=1, random_state=1).reset_index(drop=True)
        n = df_domain.shape[0]
        df_train = df_domain.iloc[:int(0.7 * n), :]
        df_test = df_domain.iloc[int(0.7 * n):, :]

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

        df_train = self._fix_df_structure(df_train)
        df_test = self._fix_df_structure(df_test)

        overlap = list(set(df_train['id']) & set(df_test['id']))
        assert (len(overlap) == 0)

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
                                license_link='https://cvgl.stanford.edu/projects/uav_data/',
                                license_requirements='None',
                                license_citation='http://svl.stanford.edu/assets/papers/ECCV16social.pdf'
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
        # Delete original director
        shutil.rmtree(self.path.joinpath('annotations'))
        shutil.rmtree(self.path.joinpath('videos'))
        shutil.rmtree(self.path.joinpath('tmp_imgs'))
        os.remove(self.path.joinpath('README'))

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
