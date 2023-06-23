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
from pathlib import Path
import pandas as pd
import shutil
from lwll_dataset_prep.logger import log


@dataclass
class mnist(BaseProcesser):
    """
    Our source data is in the form:
    mnist_png
    - training
    - - 0
    - - - img1.png
    - - - img2.png
    - - - etc.
    - - 1
    - - - imgx.png
    - - - imgy.png
    - - - etc.
        ...
    - - 9
    - - - imgx.png
    - - - imgx.png
    - - - etc.
    - testing (same as training)

    Roughly 60k in training folder and 10k in testing folder

    We will transform this into our LwLL format for image problems (See this repo's README for that format)

    """
    data_path: Path = Path('lwll_datasets')
    labels_path: Path = Path('lwll_labels')
    tar_path: Path = Path('lwll_compressed_datasets')
    _path_name: str = 'mnist'
    _task_type: str = 'image_classification'
    _urls: List[str] = field(default_factory=lambda: ['https://github.com/myleott/mnist_png/raw/master/mnist_png.tar.gz'])
    _sample_size_train: int = 5000
    _sample_size_test: int = 1000
    _valid_extensions: List[str] = field(default_factory=lambda: ['.png'])
    _k_seed: List[int] = field(default_factory=lambda: [1])

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

    def create_metafile(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                        df_train_sample: pd.DataFrame, df_test_sample: pd.DataFrame) -> None:
        p = self.labels_path.joinpath(self._path_name)
        p.joinpath(f'{self._path_name}_sample').mkdir(parents=True, exist_ok=True)
        p.joinpath(f'{self._path_name}_full').mkdir(parents=True, exist_ok=True)
        df_train_meta = pd.DataFrame(columns=['unseen_ids'])
        zsl_rows = df_train[df_train['class'] == '9']['id'].tolist()
        df_train_meta['unseen_ids'] = zsl_rows
        df_train_meta.to_feather(p.joinpath(f'{self._path_name}_full/meta_zsl_train.feather'))

        df_test_meta = pd.DataFrame(columns=['unseen_ids'])
        zsl_rows = df_test[df_test['class'] == '9']['id'].tolist()
        df_test_meta['unseen_ids'] = zsl_rows
        df_test_meta.to_feather(p.joinpath(f'{self._path_name}_full/meta_zsl_test.feather'))

        df_train_meta = pd.DataFrame(columns=['unseen_ids'])
        zsl_rows = df_train_sample[df_train_sample['class'] == '9']['id'].tolist()
        df_train_meta['unseen_ids'] = zsl_rows
        df_train_meta.to_feather(p.joinpath(f'{self._path_name}_sample/meta_zsl_train.feather'))

        df_test_meta = pd.DataFrame(columns=['unseen_ids'])
        zsl_rows = df_test_sample[df_test_sample['class'] == '9']['id'].tolist()
        df_test_meta['unseen_ids'] = zsl_rows
        df_test_meta.to_feather(p.joinpath(f'{self._path_name}_sample/meta_zsl_test.feather'))

        return

    def process(self) -> None:
        # Extract the tar
        for fname in self._fnames:
            self.extract_tar(dir_name=self._path_name, fname=fname)

        # Create our output directies
        self.setup_folder_structure(dir_name=self._path_name)

        # Create our data schema
        df_train = pd.DataFrame(columns=['id', 'class'])
        df_test = pd.DataFrame(columns=['id', 'class'])

        for _class in range(10):
            imgs = pd.DataFrame([{'id': p.name, 'class': str(_class)} for p in self.path.joinpath(f'mnist_png/training/{_class}').iterdir()])
            df_train = pd.concat([df_train, imgs])
            imgs = pd.DataFrame([{'id': p.name, 'class': str(_class)} for p in self.path.joinpath(f'mnist_png/testing/{_class}').iterdir()])
            df_test = pd.concat([df_test, imgs])

        # Create our sample subsets
        df_train_sample, df_test_sample = self.create_label_files(dir_name=self._path_name, df_train=df_train,
                                                                  df_test=df_test, samples_train=self._sample_size_train,
                                                                  samples_test=self._sample_size_test)
        self.create_metafile(df_train, df_test, df_train_sample, df_test_sample)

        # Move the raw data files
        orig_train_paths = [_p for _p in self.path.joinpath(f'mnist_png/training').glob('*/*') if _p.suffix in self._valid_extensions]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_train_paths, dest_type='train')

        orig_test_paths = [_p for _p in self.path.joinpath(f'mnist_png/testing').glob('*/*') if _p.suffix in self._valid_extensions]
        self.original_paths_to_destination(dir_name=self._path_name, orig_paths=orig_test_paths, dest_type='test')

        # Copy appropriate sample data to sample location
        self.copy_from_full_to_sample_destination(dir_name=self._path_name, df_train_sample=df_train_sample, df_test_sample=df_test_sample)

        # Create our `dataset.json` metadata file
        dataset_doc = DatasetDoc_zsl(
            name=self._path_name,
            dataset_type="image_classification",
            sample_number_of_samples_train=len(df_train_sample),
            sample_number_of_samples_test=len(df_test_sample),
            sample_number_of_classes=10,
            full_number_of_samples_train=len(df_train),
            full_number_of_samples_test=len(df_test),
            full_number_of_classes=10,
            number_of_channels=1,
            classes=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
            seen_classes=["0", "1", "2", "3", "4", "5", "6"],
            unseen_classes=["7", "8", "9"],
            language_from=None,
            language_to=None,
            sample_total_codecs=None,
            full_total_codecs=None,
            zsl_description=mnist_zsl_description,
            license_link="http://yann.lecun.com/exdb/mnist/",
            license_requirements="None",
            license_citation='[LeCun et al., 1998a] Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based learning applied to document recognition." Proceedings of the IEEE, 86(11):2278-2324, November 1998. http://yann.lecun.com/exdb/publis/index.html#lecun-98',  # noqa
        )
        self.save_dataset_metadata(dir_name=self._path_name, metadata=dataset_doc)

        # Validating sample datasets
        log.info("Validating sample datasets")
        self.validate_output_structure(name=self._path_name,
                                       orig_train_paths=orig_train_paths,
                                       orig_test_paths=orig_test_paths,
                                       classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                       train_df=df_train,
                                       test_df=df_test,
                                       sample_train_df=df_train_sample,
                                       sample_test_df=df_test_sample,
                                       metadata=dataset_doc)

        # Cleanup tar extraction intermediate
        log.info("Cleaning up extracted tar copy..")
        for f in self._fnames:
            # We assume the tar files have no '.'s in their name before `.tar.gz` or just `.tar`
            d = f.split('.')[0]
            shutil.rmtree(self.path.joinpath(d))

        log.info("Done")

    def transfer(self) -> None:
        log.info("Pushing artifacts to appropriate cloud resources...")
        self.push_data_to_cloud(dir_name=self._path_name, dataset_type='development', task_type=self._task_type)
        # self.push_data_to_cloud(dir_name=self._path_name, dataset_type='external', task_type=self._task_type)
        log.info("Done")


mnist_zsl_description = {
    "0": (
        "The smallest non-negative integer. "
        "In Arabic numerals, it is written as an elongated circle (i.e., oval) shape. "
        "May sometimes contain a diagonal line through the center to help distinguish it from the letter 'o'. "
    ),
    "1": (
        "The smallest positive integer. "
        "In Arabic numerals, it is written as a vertical line, which may sometimes be connected at the top "
        "to a short slanted line that points downwards and to the left. "
        "Visually similar to the uppercase letter 'I' in handwriting, as well as lowercase 'l'. "
        "Therefore, a horizontal line may be added at the bottom."
    ),
    "2": (
        "The number two. "
        "In Arabic numerals, it is written like the letter 'Z', but with the top horizontal line "
        "replaced with a curved line (semicircle). "
        "May sometimes look like a mirror image of the letter 's'."
    ),
    "3": (
        "The number three. "
        "In Arabic numerals, it appears like a mirrored uppercase letter 'E', but with "
        "curved edges instead of straight ones. "
        "It appears like the right half of the Arabic number '8'."
    ),
    "4": (
        "The number four. "
        "In Arabic numerals, it looks like a right triangle with the diagonal "
        "on the top left, but where the horizontal and vertical edges extend slightly beyond the intersection."
        "When written hastily, may sometimes look like the letter 'h' rotated counter-clockwise by around 180 degrees"
    ),
    "5": (
        "The number five. "
        "In Arabic numerals, it appears like a mirrored uppercase letter 'S', "
        "but with the top left curve replaced with a horizontal line that meets a vertical line "
        "at the top left corner. The bottom half is curved, similar to an 'S'."
    ),
    "6": (
        "The number six. "
        "In Arabic numerals, it appears like a reversed/vertically flipped '9'. "
        "May look like the lowercase letter 'b', but with the left vertical line replaced "
        "by a curved line pointing upwards and to the right."
    ),
    "7": (
        "The number seven. "
        "In Arabic numerals, it is written as a horizontal line that starts at the top left "
        "followed by a diagonal line starting at the top right corner that goes down to the "
        "bottom left. May sometimes have a horizontal line going through the center/middle "
        "of the diagonal part."
    ),
    "8": (
        "The number eight. "
        "In Arabic numerals, it appears like two circles stacked on top of each other, "
        "with the top one being slightly smaller. Looks like a vertical infinity symbol."
    ),
    "9": (
        "The number nine. "
        "In Arabic numerals, it appears like a '6' rotated counter-clockwise by 180 degrees, "
        "with a circle at the bottom connected at the rightmost point on the circle to a "
        "curved line that points downwards and to the left. "
        "May sometimes look like the mirror image of a lowercase letter 'p'. "
        "Also resembles to the handwritted lowercase letter 'g'."
    ),
}
