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
import os
from typing import List, Optional, Any
from pathlib import Path
import pandas as pd
import shutil
import pickle
from tqdm import tqdm
import imageio
import numpy as np
from lwll_dataset_prep.logger import log

@dataclass
class cifar100_zsl(BaseProcesser):
    """
    Our source data is in the form:
    cifar100
    - train
    - - apple
    - - - 00001.png
    - - - 00123.png
    - - - etc.
    - - aquarium_fish
    - - - 00076.png
    - - - 00210.png
    - - - etc.
            ...
    - - baby
    - - - 00612.png
    - - - 00037.png
    - - - etc.
    - test (same as training)

    50,000 in training folder and 10,000 in testing folder

    We preprocess the python pickle format of the datasets to transform to png using
    https://github.com/knjcode/cifar2png, we change the code to rename the png files using a global counter
    cmd: cifar2png <dataset> <output_dir>

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'cifar100_zsl'
    _task_type: str = 'image_classification'
    _zsl_labels: List[str] = field(default_factory=lambda: ['crab', 'beaver', 'rocket'])
    _urls: List[str] = field(default_factory=lambda: ['https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'])
    _sample_size_train: int = 5000
    _sample_size_test: int = 1000
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

        # Create our data schema
        self.df_train: pd.core.frame.DataFrame = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        self.df_test: pd.core.frame.DataFrame = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        self.df_test_zsl: pd.core.frame.DataFrame = pd.DataFrame(columns=['id', 'class', 'orig_path'])

    def create_zsl_label_files(self, dir_name: str) -> None:
        p = self.labels_path.joinpath(dir_name)
        p.joinpath(f'{dir_name}_zsl').mkdir(parents=True, exist_ok=True)
        self.df_test_zsl.reset_index(drop=True, inplace=True)
        log.info("Saving zero shot learning labels out...")
        self.df_test_zsl.to_feather(p.joinpath(f'{dir_name}_zsl/labels_test_zsl.feather'))
        return

    def unpickle(self, file: str) -> Any:
        with open(file, 'rb') as fo:
            res = pickle.load(fo, encoding='bytes')
        return res

    def create_zsl_dir(self, dir_name: str) -> None:
        log.info("Creating zero shot learning folder...")
        p = self.data_path.joinpath(f"{dir_name}").joinpath(f"{dir_name}_zsl")
        p.mkdir(parents=True, exist_ok=True)
        return

    def create_img_dirs(self, dir_name: Path) -> None:
        if not os.path.exists(os.path.join(dir_name, 'imgs', 'train')):
            os.makedirs((os.path.join(dir_name, 'imgs', 'train')))
        if not os.path.exists(os.path.join(dir_name, 'imgs', 'test')):
            os.makedirs((os.path.join(dir_name, 'imgs', 'test')))
        if not os.path.exists(os.path.join(dir_name, 'imgs', 'test_zsl')):
            os.makedirs((os.path.join(dir_name, 'imgs', 'test_zsl')))

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

    def extract_cifar100(self) -> None:
        log.info("Extracting cifar100 from compressed format...")
        meta = self.unpickle(os.path.join(self.path, 'cifar-100-python/meta'))
        fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]

        orig_path_zsl: List[Path] = []
        labels_zsl: List[str] = []
        img_ids_zsl: List[str] = []

        orig_path_train: List[Path] = []
        labels_train: List[str] = []
        img_ids_train: List[str] = []

        orig_path_test: List[Path] = []
        labels_test: List[str] = []
        img_ids_test: List[str] = []

        # Extracting Training set
        train = self.unpickle(os.path.join(self.path, 'cifar-100-python', 'train'))
        filenames = [t.decode('utf8') for t in train[b'filenames']]
        fine_labels = train[b'fine_labels']
        data = train[b'data']

        images = list()
        for d in data:
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
            image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
            image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
            images.append(image)

        for index, image in tqdm(enumerate(images)):
            filename = filenames[index]
            label = fine_labels[index]
            label = fine_label_names[label]
            if label not in self._zsl_labels:
                img_path = os.path.join(self.path, 'imgs', 'train', filename)
                imageio.imwrite(os.path.join(self.path, 'imgs', 'train', filename), image)
                img_ids_train.append(os.path.basename(img_path))
                orig_path_train.append(Path(img_path))
                labels_train.append(label)
            else:
                img_path = os.path.join(self.path, 'imgs', 'test_zsl', filename)
                imageio.imwrite(os.path.join(self.path, 'imgs', 'test_zsl', filename), image)
                img_ids_zsl.append(os.path.basename(img_path))
                orig_path_zsl.append(Path(img_path))
                labels_zsl.append(label)

        self.df_train['orig_path'] = orig_path_train
        self.df_train['id'] = img_ids_train
        self.df_train['class'] = labels_train

        # Extracting Test Set
        test = self.unpickle(os.path.join(self.path, 'cifar-100-python', 'test'))
        filenames = [t.decode('utf8') for t in test[b'filenames']]
        fine_labels = test[b'fine_labels']
        data = test[b'data']

        images = list()
        for d in data:
            image = np.zeros((32, 32, 3), dtype=np.uint8)
            image[..., 0] = np.reshape(d[:1024], (32, 32))  # Red channel
            image[..., 1] = np.reshape(d[1024:2048], (32, 32))  # Green channel
            image[..., 2] = np.reshape(d[2048:], (32, 32))  # Blue channel
            images.append(image)

        for index, image in tqdm(enumerate(images)):
            filename = filenames[index]
            label = fine_labels[index]
            label = fine_label_names[label]
            if label not in self._zsl_labels:
                img_path = os.path.join(self.path, 'imgs', 'test', filename)
                imageio.imwrite(os.path.join(self.path, 'imgs', 'test', filename), image)
                img_ids_test.append(os.path.basename(img_path))
                orig_path_test.append(Path(img_path))
                labels_test.append(label)
            else:
                img_path = os.path.join(self.path, 'imgs', 'test_zsl', filename)
                imageio.imwrite(os.path.join(self.path, 'imgs', 'test_zsl', filename), image)
                img_ids_zsl.append(os.path.basename(img_path))
                orig_path_zsl.append(Path(img_path))
                labels_zsl.append(label)

        self.df_test['orig_path'] = orig_path_test
        self.df_test['id'] = img_ids_test
        self.df_test['class'] = labels_test

        self.df_test_zsl['orig_path'] = orig_path_zsl
        self.df_test_zsl['id'] = img_ids_zsl
        self.df_test_zsl['class'] = labels_zsl

        shutil.rmtree(os.path.join(self.path, 'cifar-100-python'))

    def filter_small_cls(self) -> None:
        field = 'id'
        class_df = self.df_train[['class', field]].groupby('class').count().reset_index()
        class_rm = class_df.loc[class_df[field] < max(self._k_seed)]

        for rm in class_rm['class']:
            self.df_train = self.df_train[self.df_train['class'] != rm]
            self.df_test = self.df_test[self.df_test['class'] != rm]
        class_df = self.df_train[['class', field]].groupby('class').count().reset_index()
        class_rm = class_df.loc[class_df[field] < max(self._k_seed)]
        return

    def download(self) -> None:
        # Download
        for url, fname in zip(self._urls, self._fnames):
            self.download_data_from_url(url=url, dir_name=self._path_name, file_name=fname, overwrite=False)
            log.info("Done")

    def process(self) -> None:
        # Extract the tar
        # This dataset is preprocessed, so we do not have to untar.
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)
        self.create_zsl_dir(dir_name=self._path_name)
        self.create_img_dirs(dir_name=self.path)

        # Extracting dataset from compressed format
        self.extract_cifar100()
        self.filter_small_cls()

        # extract original paths
        orig_train_paths = self.df_train['orig_path'].tolist()
        orig_test_paths = self.df_test['orig_path'].tolist()
        orig_test_paths_zsl = self.df_test_zsl['orig_path'].tolist()

        self.df_train = self.df_train.drop(['orig_path'], axis=1)
        self.df_test = self.df_test.drop(['orig_path'], axis=1)
        self.df_test_zsl = self.df_test_zsl.drop(['orig_path'], axis=1)

        # Sanity checks
        overlap = list(set(self.df_train['id']) & set(self.df_test['id']) & set(self.df_test_zsl['id']))
        assert (len(overlap) == 0)
        assert(sorted(self.df_train['class'].unique()) == sorted(self.df_test['class'].unique()))

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=self.df_train,
                                                                  df_test=self.df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)
        self.create_zsl_label_files(dir_name=self._path_name)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train', delete_original=False)
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test', delete_original=False)
        self.original_paths_to_destination_zsl(dir_name=self._path_name, orig_paths=orig_test_paths_zsl, delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=len(df_train_sample["id"].unique()),
                                 sample_number_of_samples_test=len(df_test_sample["id"].unique()),
                                 sample_number_of_classes=len(df_train_sample["class"].unique()),
                                 full_number_of_samples_train=len(self.df_train["id"].unique()),
                                 full_number_of_samples_test=len(self.df_test["id"].unique()),
                                 full_number_of_classes=len(self.df_train["class"].unique()),
                                 number_of_channels=3,
                                 classes=list(self.df_train['class'].unique()),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://www.cs.toronto.edu/\~kriz/cifar.html',
                                 license_requirements='None',
                                 license_citation='Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009. \
                                 https://www.cs.toronto.edu/\~kriz/learning-features-2009-TR.pdf',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validate sample subsets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=list(self.df_train['class'].unique()),
                                       train_df=self.df_train, test_df=self.df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup
        os.remove(self.path.joinpath("cifar-100-python.tar.gz"))
        shutil.rmtree(self.path.joinpath("imgs"))
        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        print(self._task_type)
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
