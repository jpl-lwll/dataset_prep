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
import shutil
import zipfile
from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from typing import List
from dataclasses import dataclass, field
from pathlib import Path
from lwll_dataset_prep.logger import log


@dataclass
class covid_chestXray(BaseProcesser):
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'covid_chestXray'
    _task_type: str = 'image_classification'
    _url: str = 'raw/covid_chestXray.zip'
    _fname: str = 'covid_chestXray.zip'
    _sample_size_train: int = 200
    _sample_size_test: int = 30
    _k_seed: List[int] = field(default_factory=lambda: [1])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self.full_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train')
        self.sample_path: Path = self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train')

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)

    def get_pulmo_labels(self, label_dir: str, df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
        for img in df['id']:
            filename = img.split('.')[0]
            lb = filename.split('_')[-1]

            if (lb == "0"):
                df.loc[df['id'] == img, 'class'] = 'normal'
            elif (lb == "1"):
                df.loc[df['id'] == img, 'class'] = 'tuberculosis'
            else:
                print('Error: Invalid filename')
                exit()
        return df

    def _unzip_file(self, filepath: Path) -> None:
        log.info(f"Unzipping  {filepath}")
        with zipfile.ZipFile(filepath, 'r') as zf:
            zf.extractall(self.data_path)

        if (os.path.isdir(self.path.joinpath('__MACOSX'))):
            shutil.rmtree(self.path.joinpath('__MACOSX'))

    def download(self) -> None:
        log.info(f"Downloading dataset {self._path_name}")
        self.download_s3_url(data_path=self._url, dest=self.data_path.joinpath(self._fname))
        self._unzip_file(self.data_path.joinpath(self._fname))
        log.info("Done")

    def process(self) -> None:
        log.info(f"Processing dataset {self._path_name}")
        df1 = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        path1 = os.path.join(self.path, '1_covid-chestxray-dataset')
        ds1Metadata = pd.read_csv(os.path.join(path1, 'metadata.csv'))
        ds1Metadata = ds1Metadata[ds1Metadata['folder'] == 'images']
        df1['id'] = ds1Metadata['filename']
        df1['class'] = ds1Metadata['finding']
        df1['orig_path'] = df1['id'].apply(lambda x: os.path.join(path1, 'images', x))

        drop = ['Unknown', 'todo', 'No Finding']
        df1['class'] = df1['class'].drop(df1[df1['class'].isin(drop)].index)
        df1['class'] = df1['class'].dropna()
        df1 = df1[~df1['class'].isnull()]
        df1['class'] = df1['class'].apply(lambda x: x.replace('/', '_') if type(x) is str else x)
        df1['class'] = df1['class'].apply(lambda x: x.strip() if type(x) is str else x)

        assert (df1['orig_path'].apply(lambda x: os.path.basename(x)).all() == df1['id'].all())

        df2 = pd.DataFrame(columns=['id', 'class', 'orig_path'])
        path2 = os.path.join(self.path, '2_COVID-19_Radiography_Dataset')
        labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        imgIds: List[str] = []
        lbVals: List[str] = []
        paths: List[str] = []
        for lb in labels:
            list_files = [img for img in os.listdir(os.path.join(path2, lb, 'images')) if img.split('.')[-1] == 'png']
            imgIds += list_files
            lbVals += [lb] * len(list_files)
            paths += [os.path.join(path2, lb, 'images', x) for x in list_files]

        df2['id'] = imgIds
        df2['class'] = lbVals
        df2['orig_path'] = paths
        assert (df2['orig_path'].apply(lambda x: os.path.basename(x)).all() == df2['id'].all())
        df2 = df2.sample(frac=1, random_state=2).reset_index(drop=True)

        dfs = [df1, df2]
        df_data = pd.concat(dfs)
        assert (df1.shape[0] + df2.shape[0] == df_data.shape[0])
        df_data = df_data.sample(frac=1, random_state=2).reset_index(drop=True)

        n = df_data.shape[0]
        df_train = df_data.iloc[:int(0.7 * n), :]
        df_test = df_data.iloc[int(0.7 * n):, :]
        assert (len(list(set(df_train['id']) & set(df_test['id']))) == 0)

        # Extract original paths
        orig_paths_train = df_train['orig_path'].tolist()
        orig_paths_test = df_test['orig_path'].tolist()
        orig_paths_train = [Path(x) for x in orig_paths_train]
        orig_paths_test = [Path(x) for x in orig_paths_test]

        df_train = df_train.drop(['orig_path'], axis=1)
        df_test = df_test.drop(['orig_path'], axis=1)

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name,
                                                                  df_train=df_train,
                                                                  df_test=df_test,
                                                                  samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)
        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name,
                                           orig_paths=orig_paths_train,
                                           dest_type='train',
                                           delete_original=False)
        self.original_paths_to_destination(dir_name=self._path_name,
                                           orig_paths=orig_paths_test,
                                           dest_type='test',
                                           delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name,
                                                  df_train_sample=df_train_sample,
                                                  df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                dataset_type='image_classification',
                                sample_number_of_samples_train=len(df_train_sample),
                                sample_number_of_samples_test=len(df_test_sample),
                                sample_number_of_classes=df_train_sample['class'].nunique(),
                                full_number_of_samples_train=len(df_train),
                                full_number_of_samples_test=len(df_test),
                                full_number_of_classes=len(df_data['class'].unique()),
                                number_of_channels=3,
                                classes=list(df_data['class'].unique()),
                                language_from=None,
                                language_to=None,
                                sample_total_codecs=None,
                                full_total_codecs=None,
                                license_link="https://github.com/ieee8023/covid-chestxray-dataset/blob/master/metadata.csv\
                                              https://www.kaggle.com/tawsifurrahman/covid19-radiography-database",
                                license_requirements="(Covid Chestray)Each image has license specified in the metadata.csv \
                                                      file. Including Apache 2.0, CC BY-NC-SA 4.0, CC BY 4.0. The metadata.\
                                                      csv, scripts, and other documents are released under a CC BY-NC-SA 4.0\
                                                       license. Companies are free to perform research. Beyond that contact \
                                                       us. ; (Covid Radiography) Data files © Original Authors - Attribution\
                                                        Required for Each Data File Original Authors",
                                license_citation="(Covid Chestray)COVID-19 Image Data Collection: Prospective Predictions \
                                                  Are the Future Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten \
                                                  Roth and Tim Q Duong and Marzyeh Ghassemi \
                                                  https://github.com/ieee8023/covid-chestxray-dataset, 2020 http://arxiv.org/\
                                                  abs/2006.11988 AND COVID-19 image data collection, arXiv:2003.11597, 2020 \
                                                  Joseph Paul Cohen and Paul Morrison and Lan Dao https://github.com/ieee8023/\
                                                  covid-chestxray-dataset https://arxiv.org/abs/2003.11597 \
                                                  (Covid Radiography) M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, \
                                                  M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, \
                                                  M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 \
                                                  pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676 ; Rahman, T., \
                                                  Khandakar, A., Qiblawey, Y., Tahir, A., Kiranyaz, S., Kashem, S.B.A., \
                                                  Islam, M.T., Maadeed, S.A., Zughaier, S.M., Khan, M.S. and Chowdhury, \
                                                  M.E., 2020. Exploring the Effect of Image Enhancement Techniques on \
                                                  COVID-19 Detection using Chest X-ray Images.")  # noqa
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_paths_train,
                                       orig_test_paths=orig_paths_test,
                                       classes=list(df_data['class'].unique()),
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Removing imgs directory
        shutil.rmtree(self.path.joinpath('1_covid-chestxray-dataset'))
        shutil.rmtree(self.path.joinpath('2_COVID-19_Radiography_Dataset'))
        os.remove(self.data_path.joinpath('covid_chestXray.zip'))
        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
