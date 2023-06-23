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
import glob
from typing import List
from pathlib import Path
import pandas as pd
import os
import random
from lwll_dataset_prep.logger import log


@dataclass
class usgs_current(BaseProcesser):
    """
    Our source data is in from: thor-f5.er.usgs.gov/ngtoc/metadata/misc/topomaps_all.zip
    We manually download the PDFs then convert them to PNGs.

    There 28741/8426 images in train/test.
    We cropped the center 1000x1000 from each image an use the

    The labels are cell_name, primary_state.

    To start processing, we need to manually set up the directory like this:
    --train/
    --test/
    --historicaltopo.csv
    --ustopo_current.csv

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'usgs_current'
    _task_type: str = 'image_classification'
    _k_seed: List[int] = field(default_factory=lambda: [1])
    _urls: List[str] = field(default_factory=lambda: ['https:thor-f5.er.usgs.gov/ngtoc/metadata/misc/topomaps_all.zip'])
    _sample_size_train: int = 15
    _sample_size_test: int = 1
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))
        self.top25: List[str] = ['Corvallis_Oregon',
                                 'Webster_Massachusetts',
                                 'Santa_Fe_New_Mexico',
                                 'Kingman_Arizona',
                                 'Escondido_California',
                                 'Colfax_California',
                                 'Elkton_Maryland',
                                 'Brooklyn_New_York',
                                 'Wildcat_Mountain_Utah',
                                 'Lawrence_Massachusetts',
                                 'Augusta_Maine',
                                 'Marlborough_Massachusetts',
                                 'Fountain_Colorado',
                                 'Narcoossee_Florida',
                                 'Fish_Springs_Utah',
                                 'Babbitt_Minnesota',
                                 'Anchorage_C-6_Alaska',
                                 'Blackstone_Massachusetts',
                                 'Marquette_Michigan',
                                 'Fort_Pierce_Florida',
                                 'Los_Angeles_California',
                                 'Raquette_Lake_New_York',
                                 'Jefferson_City_Missouri',
                                 'Las_Vegas_Nevada',
                                 'Buffalo_New_York']

    def download(self) -> None:
        # Download
        log.info("Done")

    def create_metafile(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                        df_train_sample: pd.DataFrame, df_test_sample: pd.DataFrame) -> None:
        p = self.labels_path.joinpath(self._path_name)
        p.joinpath(f'{self._path_name}_sample').mkdir(parents=True, exist_ok=True)
        p.joinpath(f'{self._path_name}_full').mkdir(parents=True, exist_ok=True)

        return

    def process(self) -> None:
        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        # pre_train = 'https://thor-f5.er.usgs.gov/ngtoc/metadata/waf/htmc/geopdf/'
        pre_test = 'https://prd-tnm.s3.amazonaws.com/StagedProducts/Maps/USTopo/PDF/'
        csv_curr = pd.read_csv(os.path.join(self.path, 'ustopo_current.csv'), sep=",")
        csv_curr.columns = ['product_inventory_uuid', 'series', 'edition', 'map_name', 'primary_state', 'gnis_cell_id',
                            'westbc', 'eastbc', 'northbc', 'southbc', 'geom_wkt', 'grid_size', 'cell_type',
                            'nrn', 'nsn', 'state_list', 'county_list', 'publication_date', 'imagery_source_date',
                            'metadata_date', 'date_on_map', 'map_scale', 'page_width_inches', 'page_height_inches',
                            'product_filename', 'product_format', 'product_filesize', 'db_uuid', 'inv_uuid',
                            'product_url', 'thumbnail_url', 'sciencebase_url', 'metadata_url']
        csv_curr.drop_duplicates()
        csv_curr.replace(r'\s+', '_', regex=True, inplace=True)
        csv_curr.replace(r'\%20', '_', regex=True, inplace=True)
        csv_curr.replace(r'\%27', '\'', regex=True, inplace=True)
        csv_curr.replace(r'\%28', '(', regex=True, inplace=True)
        csv_curr.replace(r'\%29', ')', regex=True, inplace=True)

        item_list = []
        train_ids = []
        train_cates = []
        test_ids = []
        test_cates = []
        orig_train_paths = []
        orig_test_paths = []

        for _img in glob.glob(os.path.join(self.path, "current/*.png")):
            base = os.path.basename(_img)
            words = base.split('_')
            key = f"{pre_test}{words[0]}/{base.split('.')[0]}.pdf"
            res = csv_curr[csv_curr['product_url'] == key]
            map_name = res['map_name'].tolist()[0].split('_')
            map_id = ""
            for w in map_name:
                if w == 'NE' or w == 'NW' or w == 'SE' or w == 'SW':
                    break
                else:
                    map_id = f"{map_id}_{w}"
            map_id = map_id[1:]
            label = f"{map_id}_{res['primary_state'].tolist()[0]}"
            # label = res['primary_state'].tolist()[0]
            if label in self.top25:
                item_list.append((base, label, self.path.joinpath(f"current/{base}")))

        # print(f"len of item list: {len(item_list)}")
        random.Random(5).shuffle(item_list)
        split_idx = int(len(item_list) * 0.7)
        train_split = item_list[:split_idx]
        test_split = item_list[split_idx:]

        for i in train_split:
            train_ids.append(i[0])
            train_cates.append(i[1])
            orig_train_paths.append(i[2])
        for i in test_split:
            test_ids.append(i[0])
            test_cates.append(i[1])
            orig_test_paths.append(i[2])

        train_cls = set(train_cates)
        print(f"len train class: {len(train_cls)}")
        test_cls = set(test_cates)
        print(f"len test class: {len(test_cls)}")

        # Create our data schema
        df_train = pd.DataFrame({'id': train_ids, 'class': train_cates})
        print(df_train.groupby('class').count().sort_values('id', ascending=False).to_string())
        df_test = pd.DataFrame({'id': test_ids, 'class': test_cates})
        print(df_test.groupby('class').count().sort_values('id', ascending=False).to_string())

        diff = train_cls.difference(test_cls)
        for c in diff:
            df_train.drop(df_train[df_train['class'].isin(diff)].index, inplace=True)

        diff = test_cls.difference(train_cls)
        for c in diff:
            df_test.drop(df_test[df_test['class'].isin(diff)].index, inplace=True)

        # drop Virgin_Islands
        df_train.drop(df_train[df_train['class'] == 'Virgin_Islands'].index, inplace=True)
        df_test.drop(df_test[df_test['class'] == 'Virgin_Islands'].index, inplace=True)
        print(f"after filtering: {len(set(df_train['class'].tolist()))}")

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)
        self.create_metafile(df_train, df_test, df_train_sample, df_test_sample)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train', delete_original=False)
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test', delete_original=False)

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc(name=self._path_name,
                                 dataset_type='image_classification',
                                 sample_number_of_samples_train=len(df_train_sample),
                                 sample_number_of_samples_test=len(df_test_sample),
                                 sample_number_of_classes=len(set(test_cates)),
                                 full_number_of_samples_train=len(df_train),
                                 full_number_of_samples_test=len(df_test),
                                 full_number_of_classes=len(test_cls),
                                 number_of_channels=3,
                                 classes=list(test_cls),
                                 language_from=None,
                                 language_to=None,
                                 sample_total_codecs=None,
                                 full_total_codecs=None,
                                 license_link='None',
                                 license_requirements='None',
                                 license_citation='None',  # noqa
                                 )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=test_cls,
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        # Manually move train/ and test/ to persistent place
        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")
