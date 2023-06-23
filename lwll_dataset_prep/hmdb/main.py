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
import cv2
from dataclasses import dataclass, field
import glob
import os
from typing import List
from pathlib import Path
import pandas as pd
import random
import shutil
from lwll_dataset_prep.logger import log


@dataclass
class hmdb(BaseProcesser):
    """
    Our source data is in the form:
    hmdb51_org
    - brush_hair
    - - April_09_brush_hair_u_nm_np1_ba_goo_0.avi
    - - April_09_brush_hair_u_nm_np1_ba_goo_1.avi
    - - April_09_brush_hair_u_nm_np1_ba_goo_2.avi
        ...
    - dribble
    - - 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_ba_med_5.avi
    - - 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_fr_med_0.avi
    - - 10YearOldYouthBasketballStarBaller_dribble_f_cm_np1_fr_med_1.avi
        ...
    - hug
    - - -_FREE_HUGS_-_Abrazos_Gratis_www_abrazosgratis_org_hug_u_cm_np2_ba_goo_10.avi
    - - -_FREE_HUGS_-_Abrazos_Gratis_www_abrazosgratis_org_hug_u_cm_np2_ba_goo_2.avi
    - - -_FREE_HUGS_-_Abrazos_Gratis_www_abrazosgratis_org_hug_u_cm_np2_ba_goo_6.avi


    Roughly 6766 video clips in 51 folders.

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'hmdb'
    _task_type: str = 'video_classification'
    _urls: List[str] = field(default_factory=lambda: ['http://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/hmdb51_org.rar'])
    _sample_size_train: int = 1000
    _sample_size_test: int = 100
    _valid_extensions: List[str] = field(default_factory=lambda: ['.avi'])

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
        # This dataset is in rar format, so we have to manually extract it.
        # hmdb51_org is expected to be the path
        # for fname in self._fnames:
        #     self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        # Create our data schema
        df_video = pd.DataFrame(columns=['id', 'video_id', 'start_frame', 'end_frame', 'class'], dtype=object)

        classes = []
        seg_frames = {}
        cnt = 0
        frame_count = 0

        # read all video paths
        v_paths = glob.glob(f"{self.path.joinpath('hmdb51_org')}/*/*")
        random.Random(4).shuffle(v_paths)

        # create segment dirs
        for i in range(len(v_paths)):
            frame_path = self.path.joinpath(f"{i}")
            if not os.path.isdir(frame_path):
                os.makedirs(frame_path)

        for v_path in v_paths:
            _class = os.path.basename(os.path.dirname(v_path))
            classes.append(_class)
            video = cv2.VideoCapture(str(v_path))
            seg_frames[cnt] = []
            start_frame = frame_count
            success, img = video.read()
            while success:
                f_path = str(self.path.joinpath(f"{cnt}/{frame_count}.jpg"))
                cv2.imwrite(f_path, img)
                success, img = video.read()
                seg_frames[cnt].append(f"{cnt}/{frame_count}.jpg")
                frame_count += 1
            end_frame = frame_count-1
            df_record = pd.DataFrame([{'id': str(cnt), 'video_id': str(cnt), 'start_frame': start_frame, 'end_frame': end_frame, 'class': _class}])
            df_video = pd.concat([df_video, df_record])
            cnt += 1
        classes = list(set(classes))

        idx = int(len(df_video) * 0.8)
        df_train = df_video[:idx]
        df_test = df_video[idx:]

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)

        # Move the raw data files
        for d in df_train['video_id'].unique().tolist():
            d_path = self.data_path.joinpath(f'{self._path_name}/{self._path_name}_full/train/{d}')
            os.makedirs(d_path)

        for d in df_test['video_id'].unique().tolist():
            d_path = self.data_path.joinpath(f'{self._path_name}/{self._path_name}_full/test/{d}')
            os.makedirs(d_path)

        orig_train_paths = [self.path.joinpath(p) for _id in df_train['video_id'].tolist() for p in seg_frames[int(_id)]]
        new_train_paths = [p for _id in df_train['video_id'].tolist() for p in seg_frames[int(_id)]]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths,
                                           dest_type='train', delete_original=False, new_names=new_train_paths)

        orig_test_paths = [self.path.joinpath(p) for _id in df_test['video_id'].tolist() for p in seg_frames[int(_id)]]
        new_test_paths = [p for _id in df_test['video_id'].tolist() for p in seg_frames[int(_id)]]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths,
                                           dest_type='test', delete_original=False, new_names=new_test_paths)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='video_classification',
                                 sample_number_of_samples_train=len(df_train_sample),
                                 sample_number_of_samples_test=len(df_test_sample),
                                 sample_number_of_classes=51,
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=51,
                                 number_of_channels=3,
                                 classes=classes,
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads',
                                 license_requirements='None',
                                 license_citation='Kuehne, Hildegard, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and Thomas Serre. "HMDB: a large video database for human motion recognition." In 2011 International conference on computer vision, pp. 2556-2563. IEEE, 2011.',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_vc(name=self._path_name,
                         classes=classes,
                         labels_path=self.labels_path.joinpath(self._path_name),
                         metadata=dataset_doc,
                         task_type=self._task_type)

        # Cleanup tar extraction intermediate
        # log.info("Cleaning up extracted tar copy..")
        for f in ["hmdb51_org"]:
            log.info(f"removing {self.path.joinpath(f)}")
            shutil.rmtree(self.path.joinpath(f))
        for f in range(6766):
            log.info(f"removing {self.path.joinpath(str(f))}")
            shutil.rmtree(self.path.joinpath(str(f)))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")
