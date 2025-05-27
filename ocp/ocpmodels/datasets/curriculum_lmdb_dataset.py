"""
Copyright (c) Facebook, Inc. and its affiliates.
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import re
import bisect
import logging
import pickle
import warnings
from pathlib import Path
import spglib
from typing import List, Optional, TypeVar

import lmdb
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from ocpmodels.common.registry import registry
from ocpmodels.common.typing import assert_is_instance
from ocpmodels.common.utils import pyg2_data_transform
from ocpmodels.datasets._utils import rename_data_object_keys
from ocpmodels.datasets.target_metadata_guesser import guess_property_metadata
from ocpmodels.modules.transforms import DataTransforms
from denseweight import DenseWeight

T_co = TypeVar("T_co", covariant=True)


@registry.register_dataset("curriculum-lmdb")
class CurriculumDataset(Dataset[T_co]):
    metadata_path: Path
    sharded: bool
    r"""
    Args:
            config (dict): Dataset configuration
    """
    def __init__(self, config) -> None:
        super(CurriculumDataset, self).__init__()        
        self.config = config

        self.path = Path(self.config["src"])
        assert not self.path.is_file(), f'Only support directory: {self.path}'
        db_paths = sorted(self.path.glob("*.lmdb"))
        assert len(db_paths) == 1, f"LMDBs should be exactly one in '{self.path}'"

        self.metadata_path = self.path / "metadata.npz"

        self.env = self.connect_db(db_paths[0])

        # If "length" encoded as ascii is present, use that
        length_entry = self.env.begin().get("length".encode("ascii"))
        if length_entry is not None:
            self.num_samples = pickle.loads(length_entry)
        else:
            # Get the number of stores data from the number of entries in the LMDB
            self.num_samples = self.env.stat()["entries"]

        self.key_mapping = self.config.get("key_mapping", None)
        self.transforms = DataTransforms(self.config.get("transforms", {}))

        # load the dataset into memory
        self.data_objs = []
        for obj_idx in range(self.num_samples):
            # Return features.
            datapoint_pickled = (
                self.env
                .begin()
                .get(f"{obj_idx}".encode("ascii"))
            )
            data_object = pyg2_data_transform(pickle.loads(datapoint_pickled)) # torch_geometric.data object
            
            data_object.id = f"{obj_idx}"
            data_object.sid = torch.LongTensor([0])
            data_object.fid = data_object.fid = torch.LongTensor([obj_idx])

            if self.key_mapping is not None:
                # if s2ef: key_mapping is {"y": "energy", "force": "forces"}
                data_object = rename_data_object_keys(
                    data_object, self.key_mapping
                )

            data_object = self.transforms(data_object)
            self.data_objs.append(data_object)

        self.difficulty = self._compute_complexity()
        assert torch.all((self.difficulty >= 0.) & (self.difficulty <= 1.))

        # modify by panmz, 240513
        if self.config.get("weighted_sample", False) and "train" in config['src']: 
            self.sample_weights = self.compute_sample_weights()
            logging.info(f"Loaded sample weights. Config: {config}")

    def compute_sample_weights(self) -> List:
        properties = np.array(
            [data.y for data in self.data_objs]
        ).reshape(-1)
        # Define DenseWeight
        dw = DenseWeight(alpha=1.0)
        # Fit DenseWeight and get the weights for the 1000 samples
        sample_weights = dw.fit(properties).tolist()
        assert len(sample_weights) == self.num_samples, \
            f'num_samples: {self.num_samples} v.s. sampled_weights len: {len(self.sample_weights)}'
        
        return sample_weights

    def _compute_complexity(self):
        # measure difficulty by bon complexity
        '''
        dist_angle_stds = [bond_complexity(obj) for obj in self.data_objs]
        dist_stds = torch.tensor([datum[0] for datum in dist_angle_stds], dtype=torch.float32)
        angle_stds = torch.tensor([datum[1] for datum in dist_angle_stds], dtype=torch.float32)

        norm_dist_stds = (dist_stds - dist_stds.min()) / (dist_stds.max() - dist_stds.min())
        norm_angle_stds = (angle_stds - angle_stds.min()) / (angle_stds.max() - angle_stds.min())

        return (norm_dist_stds + norm_angle_stds) / 2
        '''
        properties = np.array(
            [data.y for data in self.data_objs]
        ).reshape(-1)

        dw = DenseWeight(alpha=1.0)
        sample_weights = dw.fit(properties)
        sample_weights = torch.from_numpy(sample_weights)
        # the higher the weights, the more difficulty the samples
        difficulty = (sample_weights - sample_weights.min()) / (sample_weights.max() - sample_weights.min())
        
        return difficulty

    def __len__(self) -> int:
        return len(self.data_objs)

    def __getitem__(self, idx: int) -> T_co:
        data_object = self.data_objs[idx]

        if hasattr(self, 'sample_weights'):
            data_object.weight = self.sample_weights[idx]

        return data_object

    def connect_db(self, lmdb_path: Optional[Path] = None) -> lmdb.Environment:
        env = lmdb.open(
            str(lmdb_path),
            subdir=False,
            readonly=True,
            lock=False,
            readahead=True,
            meminit=False,
            max_readers=1,
        )
        return env

    def close_db(self) -> None:
        self.env.close()

    def get_metadata(self, num_samples: int = 100):
        # This will interogate the classic OCP LMDB format to determine
        # which properties are present and attempt to guess their shapes
        # and whether they are intensive or extensive.

        # Grab an example data point
        example_pyg_data = self.__getitem__(0)

        # Check for all properties we've used for OCP datasets in the past
        props = []
        for potential_prop in [
            "y",
            "y_relaxed",
            "stress",
            "stresses",
            "force",
            "forces",
        ]:
            if hasattr(example_pyg_data, potential_prop):
                props.append(potential_prop)

        # Get a bunch of random data samples and the number of atoms
        sample_pyg = [
            self[i]
            for i in np.random.choice(
                self.__len__(), size=(num_samples,), replace=False
            )
        ]
        atoms_lens = [data.natoms for data in sample_pyg]

        # Guess the metadata for targets for each found property
        metadata = {
            "targets": {
                prop: guess_property_metadata(
                    atoms_lens, [getattr(data, prop) for data in sample_pyg]
                )
                for prop in props
            }
        }

        return metadata


def space_group_no(obj: BaseData):
    '''
    input: a pyg data containing cystal info
    '''
    # compute the space group number
    cell = obj.cell.numpy().reshape(3, 3)
    pos = obj.pos.numpy()
    atomic_number = obj.atomic_numbers.numpy()
    
    space_group = spglib.get_spacegroup((cell, pos, atomic_number))
    if space_group is None:# the structure is invalid or asymmetric
        sg_no = 231
    else:
        match = re.match(r"([A-Za-z0-9/-]+)\s?\((\d+)\)", space_group)
        if not match: 
            raise ValueError(f"Invalid spacegroup format: {space_group}")
        else:
            sg_no = int(match.group(2))

    return sg_no


def bond_complexity(obj: BaseData):
    '''
    input: atomic positions
    '''
    positions = obj.pos
    assert positions.dim() == 2
    N = positions.shape[0]  # 原子数量

    if N <= 2: return 0., 0.

    # 1. 键长计算
    distances = torch.cdist(positions, positions)  # 计算所有原子对之间的距离，形状 (N, N)
    distances = distances[torch.triu_indices(N, N, offset=1).unbind()]  # 取上三角部分（去重）
    distance_mean = torch.mean(distances).item()  # 键长均值
    distance_std = torch.std(distances).item()  # 键长标准差
    # distance_cv = (distance_std / distance_mean) if distance_mean != 0 else 0  # 键长变异系数

    # 2. 键角计算
    if N ==3:
        return distance_std, 0.

    angles = []
    indices = torch.combinations(torch.arange(N), r=3)
    
    vec1 = positions[indices[:, 1]] - positions[indices[:, 0]]  # j - i
    vec2 = positions[indices[:, 2]] - positions[indices[:, 0]]  # k - i
    cos_theta = F.cosine_similarity(vec1, vec2)  # 计算夹角的cosine值
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0) # avoid NaN when compute arccos
    
    # 使用acos计算角度，转换为角度（度数）
    angles = torch.acos(cos_theta) * 180 / np.pi
    angle_mean = torch.mean(angles).item()  # 键角均值
    angle_std = torch.std(angles).item()  # 键角标准差
    # angle_cv = (angle_std / angle_mean) if angle_mean != 0 else 0  # 键角变异系数

    return distance_std, angle_std
    