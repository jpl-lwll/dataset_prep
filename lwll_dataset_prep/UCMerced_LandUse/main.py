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

from lwll_dataset_prep.dataset_scripts.schema import DatasetDoc_zsl
from lwll_dataset_prep.dataset_scripts.process_interface import BaseProcesser
from dataclasses import dataclass, field
from typing import List
import glob
import os
import pathlib
from pathlib import Path
import pandas as pd
import random
import shutil
from lwll_dataset_prep.logger import log

@dataclass
class UCMerced_LandUse(BaseProcesser):
    """
    Source data: http://weegee.vision.ucmerced.edu/datasets/landuse.html
    100 images in each of the 21 categories

    Folder structure:
    storage_tank/storage_tank_700.jpg
    storage_tank/storage_tank_699.jpg
    storage_tank/storage_tank_698.jpg
    ...

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'UCMerced_LandUse'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['UCMerced_LandUse.zip'])
    _sample_size_train: int = 500
    _sample_size_test: int = 500
    _valid_extensions: List[str] = field(default_factory=lambda: ['.jpg'])

    def __post_init__(self) -> None:
        self.path: Path = self.data_path.joinpath(self._path_name)
        self._fnames: List[str] = [u.split('/')[-1] for u in self._urls]
        self.full_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_full/train'))
        self.sample_path = str(self.data_path.joinpath(self._path_name).joinpath(f'{self._path_name}_sample/train'))
        self.unseen_classes = [
            "agricultural",
            "dense_residential",
            "freeway",
            "mobile_home_park",
            "river",
        ]
        self.seen_classes = [
            "airplane",
            "baseball_diamond",
            "beach",
            "buildings",
            "chaparral",
            "forest",
            "golf_course",
            "harbor",
            "intersection",
            "medium_residential",
            "overpass",
            "parking_lot",
            "runway",
            "sparse_residential",
            "storage_tanks",
            "tennis_court",
        ]

    def download(self) -> None:
        # Download
        log.info("we have to sign in and download this dataset manually: http://weegee.vision.ucmerced.edu/datasets/landuse.html")
        log.info("Done")

    def create_metafile(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_train_sample: pd.DataFrame,
        df_test_sample: pd.DataFrame,
    ) -> None:
        p = self.labels_path.joinpath(self._path_name)
        p.joinpath(f"{self._path_name}_sample").mkdir(parents=True, exist_ok=True)
        p.joinpath(f"{self._path_name}_full").mkdir(parents=True, exist_ok=True)

        # create a dataframe with one column: unseen_ids
        # 4 dataframes in total: one for train, one for test,
        # one for train_sample, one for test_sample
        # we can map these ids back to their respective classes from labels_train/test.feather
        def _create_metadata_df(df, is_train, is_sample):
            unseen_ids = df[df["class"].isin(self.unseen_classes)]["id"].tolist()
            df_meta = pd.DataFrame({"unseen_ids": unseen_ids})
            # write df_meta to file
            full_sample_str = "sample" if is_sample else "full"
            meta_filename = "meta_zsl_train" if is_train else "meta_zsl_test"
            filepath = p.joinpath(
                f"{self._path_name}_{full_sample_str}/{meta_filename}.feather"
            )
            df_meta.to_feather(filepath)

        _create_metadata_df(df_train, True, False)
        _create_metadata_df(df_test, False, False)
        _create_metadata_df(df_train_sample, True, True)
        _create_metadata_df(df_test_sample, False, True)

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_zip(dir_name=self._path_name, fname=fname)

        # Create our output directories
        self.setup_folder_structure(dir_name=self._path_name)
        orig_classes = [os.path.basename(c) for c in glob.glob(os.path.join(self.path, "UCMerced_LandUse/images/*"))]
        classes = [class_names[c] for c in orig_classes]

        orig_paths = []
        cnt = 0
        for _img in glob.glob(os.path.join(self.path, "UCMerced_LandUse/images/*/*")):
            orig_paths.append(pathlib.Path(_img))

        random.Random(5).shuffle(orig_paths)
        split_idx = int(len(orig_paths) * 0.7)
        orig_train_paths = orig_paths[:split_idx]
        orig_test_paths = orig_paths[split_idx:]

        training_ids = []
        testing_ids = []

        pth_to_id = {}
        id_to_pth = {}
        cnt = 0

        for p in orig_train_paths:
            cnt += 1
            new_id = f'img_{cnt}.jpg'
            pth_to_id[p] = new_id
            id_to_pth[new_id] = p
            training_ids.append(new_id)

        for p in orig_test_paths:
            cnt += 1
            new_id = f'img_{cnt}.jpg'
            pth_to_id[p] = new_id
            id_to_pth[new_id] = p
            testing_ids.append(new_id)

        # Create our data schema
        df_train = pd.DataFrame({'id': training_ids, 'class': [class_names[os.path.basename(os.path.dirname(id_to_pth[i]))] for i in training_ids]})
        df_test = pd.DataFrame({'id': testing_ids, 'class': [class_names[os.path.basename(os.path.dirname(id_to_pth[i]))] for i in testing_ids]})

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test, many_to_one=False)

        # Create ZSL metadata (feather) file - list of files that belong to unseen classes
        self.create_metafile(df_train, df_test, df_train_sample, df_test_sample)

        # Move the raw data files
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train',
                                           delete_original=False, new_names=df_train['id'].unique().tolist())

        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test',
                                           delete_original=False, new_names=df_test['id'].unique().tolist())

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc_zsl(
            name=self._path_name,
            dataset_type=self._task_type,
            sample_number_of_samples_train=len(df_train_sample["id"].unique()),
            sample_number_of_samples_test=len(df_test_sample["id"].unique()),
            sample_number_of_classes=len(classes),
            full_number_of_samples_train=len(df_train["id"].unique()),
            full_number_of_samples_test=len(df_test["id"].unique()),
            full_number_of_classes=len(classes),
            number_of_channels=3,
            classes=classes,
            unseen_classes=self.unseen_classes,
            seen_classes=self.seen_classes,
            language_from=None,
            language_to=None,
            sample_total_codecs=None,
            full_total_codecs=None,
            zsl_description=ucmerced_zsl_description,  # defined below
            license_link="None",
            license_requirements="None",
            license_citation='Yang, Yi, and Shawn Newsam. "Bag-of-visual-words and spatial extensions for land-use classification." In Proceedings of the 18th SIGSPATIAL international conference on advances in geographic information systems, pp. 270-279. 2010.',  # noqa
        )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validate sample subsets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=classes,
                                       train_df=df_train, test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        os.remove(self.path.joinpath("UCMerced_LandUse.zip"))
        shutil.rmtree(self.path.joinpath("UCMerced_LandUse"))

        log.info("Done")

        return

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        log.info("Done")


ucmerced_zsl_description = {
    "agricultural": (
        "An area of land used for farming, characterized by fields of crops or"
        " pastureland. From above, agricultural areas may appear as large, rectangular"
        " or irregularly shaped patches of land with a variety of colors depending on"
        " the type of crops being grown. They may also feature roads, irrigation"
        " systems, and other structures related to farming."
    ),
    "airplane": (
        "A vehicle designed for air travel that is propelled through the air by one or"
        " more jet engines or propellers. From above, airplanes may appear as long,"
        " narrow objects with wings mounted on the top of the fuselage. They may be"
        " various colors and may have identifying markings such as logos or tail"
        " numbers."
    ),
    "baseball_diamond": (
        "A field where the game of baseball is played, consisting of a diamond-shaped"
        " infield with bases at each corner and an outfield. From above, a baseball"
        " diamond may appear as a small, oval-shaped field with distinctive markings on"
        " the grass and surrounding fences or walls."
    ),
    "beach": (
        "A strip of land along the edge of a body of water, typically a sea or ocean,"
        " that is covered by sand or small rocks. From above, a beach may appear as a"
        " wide, light-colored strip of land adjacent to a darker body of water, with a"
        " line of vegetation marking the border between the two. It may also feature"
        " structures such as boardwalks, piers, or beach houses."
    ),
    "buildings": (
        "Structures made by humans for a variety of purposes, such as living, working,"
        " or storing goods. From above, buildings may appear as rectangular or"
        " square-shaped objects of various sizes and heights, with flat or sloping"
        " roofs and walls of various materials. They may be surrounded by roads,"
        " parking lots, or other structures."
    ),
    "chaparral": (
        "A type of vegetation found in regions with a Mediterranean climate,"
        " characterized by dense, spiky shrubs and dry, grassy areas. From above,"
        " chaparral may appear as a dense, textured area with a mix of green and brown"
        " colors, often found on hillsides or in areas with poor soil. It may be"
        " interspersed with other types of vegetation or land uses."
    ),
    "dense_residential": (
        "An area characterized by a high concentration of houses, apartments, or other"
        " buildings used for residential purposes. From above, a dense residential area"
        " may appear as a cluster of closely packed buildings, with a higher density of"
        " structures compared to a sparse residential area. It may also feature roads,"
        " sidewalks, and other infrastructure associated with urban living."
    ),
    "forest": (
        "An area characterized by a dense growth of trees, often covering large areas"
        " of land. From above, a forest may appear as a large, green area with a"
        " uniform canopy of tree tops and possibly a network of roads or trails. It may"
        " also feature bodies of water, clearings, or other features."
    ),
    "freeway": (
        "A type of divided highway with multiple lanes of traffic in each direction,"
        " designed for high-speed travel. From above, a freeway may appear as a long,"
        " straight or winding stretch of pavement with multiple lanes bordered by a"
        " median or barriers. It may be surrounded by other infrastructure such as"
        " overpasses, exits, and interchanges."
    ),
    "golf_course": (
        "A large, grassy area where the game of golf is played, featuring a series of"
        " holes with associated greens, tees, and bunkers. From above, a golf course"
        " may appear as a sprawling, green area with a series of small, circular or"
        " oval-shaped features surrounded by trees or other vegetation. It may also"
        " feature other structures such as clubhouses, pro shops, or cart paths."
    ),
    "harbor": (
        "A sheltered area of water where ships can anchor, typically located along a"
        " coast or at the mouth of a river. From above, a harbor may appear as a small"
        " or large body of water with a network of docks, piers, and other structures"
        " used for mooring and loading ships. It may also feature other infrastructure"
        " such as cranes, warehouses, and offices."
    ),
    "intersection": (
        "A place where two or more roads or streets meet and cross each other. From"
        " above, an intersection may appear as a point where multiple roads converge,"
        " often marked by traffic signals or road signs. It may also feature other"
        " infrastructure such as pedestrian crossings, turn lanes, or median islands."
    ),
    "medium_residential": (
        "An area characterized by a moderate concentration of houses, apartments, or"
        " other buildings used for residential purposes. From above, a medium"
        " residential area may appear as a group of houses or apartment buildings with"
        " a moderate density of structures, intermediate between a dense and a sparse"
        " residential area. It may also feature roads, sidewalks, and other"
        " infrastructure associated with urban living."
    ),
    "mobile_home_park": (
        "An area where mobile homes or trailers are placed on individual plots of land"
        " for use as permanent or semi-permanent residences. From above, a mobile home"
        " park may appear as a collection of small, rectangular or square-shaped"
        " structures with a uniform layout and possibly shared amenities such as"
        " playgrounds or pools."
    ),
    "overpass": (
        "A bridge or elevated structure that carries a road or other transportation"
        " route over another road or feature. From above, an overpass may appear as a"
        " long, narrow structure with a roadway on top and support columns or pillars"
        " below. It may span over a roadway, railway, or other feature."
    ),
    "parking_lot": (
        "A large, open area used for parking vehicles, typically found at airports,"
        " shopping malls, or other large facilities. From above, a parking lot may"
        " appear as a large, rectangular or square-shaped area with a grid of marked"
        " spaces for parking, often surrounded by roads or other infrastructure. It may"
        " also feature other structures such as ticket booths, pay stations, or"
        " pedestrian walkways."
    ),
    "river": (
        "A natural watercourse, typically freshwater, flowing towards an ocean, sea, or"
        " another river. From above, a river may appear as a thin, sinuous line of blue"
        " or green color, often meandering through a landscape and potentially"
        " featuring branches, tributaries, or floodplains. It may be bordered by"
        " vegetation, settlements, or other features."
    ),
    "runway": (
        "A strip of paved surface used for taking off and landing airplanes at an"
        " airport. From above, a runway may appear as a long, straight or slightly"
        " curved stretch of pavement with a distinctive surface and markings. It may be"
        " surrounded by other airport infrastructure such as taxiways, hangars, and"
        " terminal buildings."
    ),
    "sparse_residential": (
        "An area characterized by a low concentration of houses, apartments, or other"
        " buildings used for residential purposes. From above, a sparse residential"
        " area may appear as a group of houses or apartment buildings with a low"
        " density of structures, intermediate between a dense and a medium residential"
        " area. It may also feature roads, sidewalks, and other infrastructure"
        " associated with urban living."
    ),
    "storage_tanks": (
        "Large, cylindrical or rectangular containers used for storing liquids or"
        " gases, often made of metal or concrete. From above, storage tanks may appear"
        " as large, round or rectangular objects with a uniform shape and color, often"
        " found in industrial areas or near ports or terminals. They may be surrounded"
        " by other infrastructure such as pipelines, valves, or fencing."
    ),
    "tennis_court": (
        "A tennis court is a rectangular or oval-shaped area where the game of tennis"
        " is played. It is marked with a series of lines and features a net stretched"
        " across the center. From above, a tennis court may appear as a small,"
        " rectangular or oval-shaped area with distinctive white or green lines marking"
        " the playing surface. It may be surrounded by other features such as fencing,"
        " lights, or seating."
    ),
}

class_names = {
    "agricultural": "agricultural",
    "denseresidential": "dense_residential",
    "mediumresidential": "medium_residential",
    "sparseresidential": "sparse_residential",
    "airplane": "airplane",
    "forest": "forest",
    "mobilehomepark": "mobile_home_park",
    "storagetanks": "storage_tanks",
    "baseballdiamond": "baseball_diamond",
    "freeway": "freeway",
    "overpass": "overpass",
    "tenniscourt": "tennis_court",
    "beach": "beach",
    "golfcourse": "golf_course",
    "parkinglot": "parking_lot",
    "buildings": "buildings",
    "harbor": "harbor",
    "river": "river",
    "chaparral": "chaparral",
    "intersection": "intersection",
    "runway": "runway"
}
