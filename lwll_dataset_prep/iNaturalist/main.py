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
import json
import os
from typing import List
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log

@dataclass
class iNaturalist(BaseProcesser):
    """

    This dataset contains 1,010 species, with a combined training and validation set of 268,243 images
    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'iNaturalist'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['/mnt/lwll-local/zzhang2/iNaturalist/train_val2019.tar.gz',
                                                      '/mnt/lwll-local/zzhang2/iNaturalist/train2019.json',
                                                      '/mnt/lwll-local/zzhang2/iNaturalist/val2019.json'])
    _sample_size_train: int = 5000
    _sample_size_test: int = 1000
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))

    def download(self) -> None:
        # Download
        # This dataset is available as a whole package on Kaggle, so we have to download it separately
        self.setup_folder_structure(dir_name=self._path_name)
        for url, fname in zip(self._urls, self._fnames):
            shutil.copy(url, self.data_path.joinpath(self._path_name).joinpath(fname))
            log.info("done")

    def process(self) -> None:
        # Extract the tar
        # This dataset is preprocessed, so we do not have to untar.
        for fname in self._fnames:
            if fname.endswith(".tar.gz"):
                self.extract_tar(dir_name=self._path_name, fname=fname)

        train_split = self.parse_json(self.data_path.joinpath(self._path_name).joinpath(f"train2019.json"))
        test_split = self.parse_json(self.data_path.joinpath(self._path_name).joinpath(f"val2019.json"))

        train_ids = []
        train_cates = []
        test_ids = []
        test_cates = []
        orig_train_paths = []
        orig_test_paths = []

        for p in train_split:
            words = p.split("/")
            cate = f"{words[1]}-{words[2]}"
            train_ids.append(words[3])
            train_cates.append(cate)
            orig_train_paths.append(self.path.joinpath(p))

        for p in test_split:
            words = p.split("/")
            cate = f"{words[1]}-{words[2]}"
            test_ids.append(words[3])
            test_cates.append(cate)
            orig_test_paths.append(self.path.joinpath(p))

        classes = list(set(train_cates))

        # Create our data schema
        df_train = pd.DataFrame({'id': train_ids, 'class': train_cates})
        df_test = pd.DataFrame({'id': test_ids, 'class': test_cates})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train')
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test')

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=len(df_train_sample["id"].unique()),
                                 sample_number_of_samples_test=len(df_test_sample["id"].unique()),
                                 sample_number_of_classes=len(list(set(classes))),
                                 full_number_of_samples_train=len(df_train["id"].unique()),
                                 full_number_of_samples_test=len(df_test["id"].unique()),
                                 full_number_of_classes=len(list(set(classes))),
                                 number_of_channels=3,
                                 classes=list(set(classes)),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='https://www.kaggle.com/c/inaturalist-2019-fgvc6',
                                 license_requirements='None',
                                 license_citation='None',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validate sample subsets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=list(set(classes)),
                                       train_df=df_train, test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in ['train2019.json', 'val2019.json', 'train_val2019.tar.gz']:
            os.remove(self.path.joinpath(f))
        shutil.rmtree(self.path.joinpath("train_val2019"))
        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        print(self._task_type)
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")

    def parse_json(self, p: Path) -> List[str]:
        l: List = []
        with open(p, 'r') as file:
            jfile = json.load(file)
        images = jfile["images"]
        for img in images:
            l.append(img["file_name"])
        return l
