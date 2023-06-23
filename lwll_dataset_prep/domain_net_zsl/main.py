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
from typing import List, Tuple, Optional
from pathlib import Path
import pandas as pd
import os
import pickle
import shutil
from lwll_dataset_prep.logger import log

@dataclass
class domain_net_zsl(BaseProcesser):
    """
    Our source data has six distinct domains and is in the form:
    clipart
    - zigzag
    - - img1.png
    - - img2.png
    - - etc.
    - zebra
    - - imgx.png
    - - imgy.png
    - - etc.
                  ...
    - airplane
    - - imgx.png
    - - imgx.png
    - - etc.
    quickdraw
    - zigzag
    - - img1.png
    - - img2.png
    - - etc.
    - zebra
    - - imgx.png
    - - imgy.png
    - - etc.
                  ...
    - airplane
    - - imgx.png
    - - imgx.png
    - - etc.
    quickdraw_train.txt
    etc.

    0.6M images of 345 object categories from 6 domains: Clipart, Infograph, Painting, Quickdraw, Photo, Sketch

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _task_type: str = "image_classification"
    _path_name: str = 'domain_net_zsl'
    _zsl_labels: List[str] = field(default_factory=lambda: ['ambulance', 'shark', 'castle'])
    _k_seed: List[int] = field(default_factory=lambda: [1])
    _urls: List[str] = field(default_factory=lambda: ['http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/painting.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/quickdraw_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/painting_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/infograph_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/txt/clipart_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/real_test.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_train.txt',
                                                      'http://csr.bu.edu/ftp/visda/2019/multi-source/txt/sketch_test.txt'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]

    def create_zsl_label_files(self, dir_name: str, df_test_zsl: pd.DataFrame, many_to_one: bool = False) -> None:
        p = self.labels_path.joinpath(dir_name)
        p.joinpath(f'{dir_name}_zsl').mkdir(parents=True, exist_ok=True)
        df_test_zsl.reset_index(drop=True, inplace=True)
        log.info("Saving zero shot learning labels out...")
        df_test_zsl.to_feather(p.joinpath(f'{dir_name}_zsl/labels_test_zsl.feather'))
        return

    def filter_small_cls(self, df_train: pd.core.frame.DataFrame, df_test: pd.core.frame.DataFrame) -> Tuple[pd.core.frame.DataFrame, pd.core.frame.DataFrame]:
        field = 'id'
        class_df = df_train[['class', field]].groupby('class').count().reset_index()
        class_rm = class_df.loc[class_df[field] < max(self._k_seed)]

        for rm in class_rm['class']:
            df_train = df_train[df_train['class'] != rm]
            df_test = df_test[df_test['class'] != rm]
        class_df = df_train[['class', field]].groupby('class').count().reset_index()
        class_rm = class_df.loc[class_df[field] < max(self._k_seed)]
        return df_train, df_test

    def original_paths_to_destination_zsl(self, dir_name: str, orig_paths: List[Path],
                                          delete_original: bool = True, new_names: Optional[List[str]] = None) -> None:
        """
        new_names argument was added in order to accomodate very dumb format of face_detection dataset where mappings were messed up and
        new ids had to be generated
        """
        p = self.data_path.joinpath(dir_name)
        log.info(f"Moving images for zero shot learning into place...")
        for _idx, _p in enumerate(orig_paths):
            name = _p.name if new_names is None else new_names[_idx]
            dest = p.joinpath(f'{dir_name}_zsl/{name}')
            if delete_original:
                shutil.move(str(_p), str(dest))
            else:
                shutil.copy(str(_p), str(dest))

    def create_zsl_dir(self, dir_name: str) -> None:
        log.info("Creating zero shot learning folder...")
        p = self.data_path.joinpath(f"{dir_name}").joinpath(f"{dir_name}_zsl")
        p.mkdir(parents=True, exist_ok=True)
        return

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract only the zip files
        for fname in self._fnames:
            self.extract_zip(dir_name=self._path_name, fname=fname)

        domains = set([_p for _p in self.path.iterdir() if _p.suffix not in ['.zip', '.txt'] and os.path.isdir(_p)])
        print(domains)

        all_classes: List[str] = []
        all_classes = pickle.load(open(os.path.join('lwll_dataset_prep', 'domain_net_zsl', 'classes.p'), 'rb'))

        # create dirs for all
        self.setup_folder_structure(dir_name=f"{self._path_name}-all")
        self.create_zsl_dir(dir_name=f"{self._path_name}-all")
        global_training_ids = []
        global_training_class = []
        global_testing_ids = []
        global_testing_class = []
        global_testing_zsl_ids = []
        global_testing_zsl_class = []

        for domain_path in domains:
            domain = os.path.basename(domain_path)
            self.setup_folder_structure(dir_name=f"{self._path_name}-{domain}")
            self.create_zsl_dir(dir_name=f"{self._path_name}-{domain}")
            training_ids = []
            training_class = []
            testing_ids = []
            testing_class = []
            testing_zsl_ids = []
            testing_zsl_class = []

            with open(self.path.joinpath(f'{domain}_train.txt')) as f:
                for line in f:
                    id_path = line.split()[0]
                    t_class = id_path.split('/')[-2]
                    if t_class not in self._zsl_labels:
                        training_ids.append(id_path.split('/')[-1])
                        training_class.append(t_class)
                    else:
                        testing_zsl_ids.append(id_path.split('/')[-1])
                        testing_zsl_class.append(t_class)

            with open(self.path.joinpath(f'{domain}_test.txt')) as f:
                for line in f:
                    id_path = line.split()[0]
                    t_class = id_path.split('/')[-2]
                    if t_class not in self._zsl_labels:
                        testing_ids.append(id_path.split('/')[-1])
                        testing_class.append(id_path.split('/')[-2])
                    else:
                        testing_zsl_ids.append(id_path.split('/')[-1])
                        testing_zsl_class.append(t_class)

            # Create our data schema
            df_train = pd.DataFrame({'id': training_ids, 'class': training_class})
            df_test = pd.DataFrame({'id': testing_ids, 'class': testing_class})
            df_test_zsl = pd.DataFrame({'id': testing_zsl_ids, 'class': testing_zsl_class})
            df_train, df_test = self.filter_small_cls(df_train, df_test)
            sample_size_train = int(len(df_train) * 0.1)
            sample_size_test = int(len(df_test) * 0.1)
            df_train_sample, df_test_sample = self.create_label_files(dir_name=f"{self._path_name}-{domain}", df_train=df_train,
                                                                      df_test=df_test, samples_train=sample_size_train,
                                                                      samples_test=sample_size_test)
            self.create_zsl_label_files(dir_name=f"{self._path_name}-{domain}", df_test_zsl=df_test_zsl)

            orig_train_paths = []
            orig_test_paths = []
            orig_test_zsl_paths = []
            for index, row in df_train.iterrows():
                orig_train_paths.append(self.path.joinpath(f'{domain}/{row["class"]}/{row["id"]}'))

            for index, row in df_test.iterrows():
                orig_test_paths.append(self.path.joinpath(f'{domain}/{row["class"]}/{row["id"]}'))

            for index, row in df_test_zsl.iterrows():
                orig_test_zsl_paths.append(self.path.joinpath(f'{domain}/{row["class"]}/{row["id"]}'))

            self.original_paths_to_destination(dir_name=f"{self._path_name}-{domain}", orig_paths=orig_train_paths, dest_type='train', delete_original=False)
            self.original_paths_to_destination(dir_name=f"{self._path_name}-{domain}", orig_paths=orig_test_paths, dest_type='test', delete_original=False)
            self.original_paths_to_destination_zsl(dir_name=f"{self._path_name}-{domain}", orig_paths=orig_test_zsl_paths, delete_original=False)

            # copy images to domain_net-all/all-full/
            self.original_paths_to_destination(dir_name=f"{self._path_name}-all", orig_paths=orig_train_paths, dest_type='train', delete_original=False)
            self.original_paths_to_destination(dir_name=f"{self._path_name}-all", orig_paths=orig_test_paths, dest_type='test', delete_original=False)
            self.original_paths_to_destination_zsl(dir_name=f"{self._path_name}-all", orig_paths=orig_test_zsl_paths, delete_original=False)

            self.copy_from_full_to_sample_destination(dir_name=f"{self._path_name}-{domain}", df_train_sample=df_train_sample,
                                                      df_test_sample=df_test_sample)

            # Create our `dataset.json` metadata file
            dataset_doc = DatasetDoc(name=f"{self._path_name}-{domain}",
                                    dataset_type='image_classification',
                                    sample_number_of_samples_train=len(df_train_sample),
                                    sample_number_of_samples_test=len(df_test_sample),
                                    sample_number_of_classes=len(df_train_sample["class"].unique()),
                                    full_number_of_samples_train=len(df_train),
                                    full_number_of_samples_test=len(df_test),
                                    full_number_of_classes=len(df_train["class"].unique()),
                                    number_of_channels=3,
                                    classes=all_classes,
                                    language_from=None,
                                    language_to=None,
                                    sample_total_codecs=None,
                                    full_total_codecs=None,
                                    license_link='http://ai.bu.edu/M3SDA/',
                                    license_requirements='None',
                                    license_citation='{{@article{peng2018moment,title={Moment Matching for Multi-Source Domain Adaptation},author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},journal={arXiv preprint arXiv:1812.01754},year={2018}}}}',  # noqa
                                    )

            self.save_dataset_metadata(dir_name=f"{self._path_name}-{domain}", metadata=dataset_doc)

            # get training/testing id/class for all
            global_training_ids.extend(training_ids)
            global_training_class.extend(training_class)
            global_testing_ids.extend(testing_ids)
            global_testing_class.extend(testing_class)
            global_testing_zsl_ids.extend(testing_zsl_ids)
            global_testing_zsl_class.extend(testing_zsl_class)

        print(len(global_training_ids), len(global_training_class), len(global_testing_ids), len(global_testing_class))
        # Create data schema for all
        df_train = pd.DataFrame({'id': global_training_ids, 'class': global_training_class})
        df_test = pd.DataFrame({'id': global_testing_ids, 'class': global_testing_class})
        df_train, df_test = self.filter_small_cls(df_train, df_test)
        df_test_zsl = pd.DataFrame({'id': global_testing_zsl_ids, 'class': global_testing_zsl_class})
        sample_size_train = int(len(df_train) * 0.1)
        sample_size_test = int(len(df_test) * 0.1)
        df_train_sample, df_test_sample = self.create_label_files(dir_name=f"{self._path_name}-all", df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=sample_size_train,
                                                                  samples_test=sample_size_test)
        self.create_zsl_label_files(dir_name=f"{self._path_name}-{domain}", df_test_zsl=df_test_zsl)

        self.copy_from_full_to_sample_destination(dir_name=f"{self._path_name}-all", df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        dataset_doc = DatasetDoc(name=f"{self._path_name}-all",
                                dataset_type='image_classification',
                                sample_number_of_samples_train=len(df_train_sample),
                                sample_number_of_samples_test=len(df_test_sample),
                                sample_number_of_classes=len(df_train_sample["class"].unique()),
                                full_number_of_samples_train=len(df_train),
                                full_number_of_samples_test=len(df_test),
                                full_number_of_classes=len(df_train["class"].unique()),
                                number_of_channels=3,
                                classes=all_classes,
                                language_from=None,
                                language_to=None,
                                sample_total_codecs=None,
                                full_total_codecs=None,
                                license_link='http://ai.bu.edu/M3SDA/',
                                license_requirements='None',
                                license_citation='{{@article{peng2018moment,title={Moment Matching for Multi-Source Domain Adaptation},author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},journal={arXiv preprint arXiv:1812.01754},year={2018}}}}',  # noqa
                                )

        self.save_dataset_metadata(dir_name=f"{self._path_name}-all", metadata=dataset_doc)

        # Cleanup zip extraction intermediate
        log.info("Cleaning up extracted zip copy..")
        shutil.rmtree(self.path)

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        domains: List[str] = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", "all"]
        for domain in domains:
            print(f"{self._path_name}-{domain}")
            self.push_data_to_cloud(dir_name=f"{self._path_name}-{domain}", dataset_type='development', task_type=self._task_type)
        log.info("Done")
