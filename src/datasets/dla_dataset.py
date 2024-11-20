import numpy as np
import torch
from tqdm.auto import tqdm
import copy
import os

from src.datasets.base_dataset import BaseDataset
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DlaDataset(BaseDataset):
    def __init__(
        self, dataset_path, dataset_length, audio_ref, video_ref, refs_cnt, name="train", override=False, *args, **kwargs
    ):
        index_path = ROOT_PATH / dataset_path / "index" / name / "index.json"

        if index_path.exists() and not override:
            index = read_json(str(index_path))
        else:
            index = self._create_index(dataset_path, dataset_length, audio_ref, video_ref, refs_cnt, name)

        super().__init__(index, *args, **kwargs)

    def _create_index(self, dataset_path, dataset_length, audio_ref, video_ref, refs_cnt, name):
        index_path = ROOT_PATH / dataset_path / "index" / name
        audio_path = ROOT_PATH / dataset_path / "audio" / name
        video_path = ROOT_PATH / dataset_path / "mouths_embs"

        data_names = [path.split('.')[0] for path in os.listdir(audio_path / "mix")]
        index_path.mkdir(exist_ok=True, parents=True)
    
        print("Creating Index for dataset")
        index = []
        dataset_length = dataset_length if dataset_length is not None else len(data_names)
        for i in tqdm(range(dataset_length)):
            data_name = data_names[i]
            s1_name = data_name.split('_')[0]
            s2_name = data_name.split('_')[1]

            data_sample = {
                "mix": str(audio_path / "mix" / f"{data_name}.wav"),
            }

            if refs_cnt == 2:
                if audio_ref:
                    data_sample.update({
                        "s1": str(audio_path / "s1" / f"{data_name}.wav"),
                        "s2": str(audio_path / "s2" / f"{data_name}.wav"),
                    })
                if video_ref:
                    data_sample.update({
                        "vid1": str(video_path / f"{s1_name}.npz"),
                        "vid2": str(video_path / f"{s2_name}.npz")
                    })
                index.append(data_sample)
            else:
                data_sample1 = copy.deepcopy(data_sample)
                data_sample2 = copy.deepcopy(data_sample)
                
                if audio_ref:
                    data_sample1.update({
                        "s1": str(audio_path / "s1" / f"{data_name}.wav"),
                    })
                    data_sample2.update({
                        "s1": str(audio_path / "s2" / f"{data_name}.wav"),
                    })
                if video_ref:
                    data_sample1.update({
                        "vid1": str(video_path / f"{s1_name}.npz")
                    })
                    data_sample2.update({
                        "vid1": str(video_path / f"{s2_name}.npz")
                    })
                index.append(data_sample1)
                index.append(data_sample2)

        write_json(index, str(index_path / "index.json"))
        return index
