import warnings

import hydra
import torch, torchaudio
from hydra.utils import instantiate
from tqdm import tqdm
import os.path

from src.utils.io_utils import ROOT_PATH
from src.metrics.tracker import MetricTracker

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="calc_metrics")
def main(config):
    """
    Main script for metrics calculation. Instantiates metrics.
    log and save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """

    if config.metrics_calculator.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.metrics_calculator.device


    metrics = instantiate(config.metrics)
    
    target_path = ROOT_PATH / config.metrics_calculator.target_path / "audio"          # directory 
    preds_path =  ROOT_PATH / config.metrics_calculator.preds_path

    print("target_path:", target_path)
    print("preds_path", preds_path)

    evaluation_metrics = MetricTracker(
        *[m.name for m in metrics["inference"]],
        writer=None,
    )
    evaluation_metrics.reset()

    metric_results = process_audio_files(target_path, preds_path, metrics, evaluation_metrics)

    metric_names = []
    metrics_vals = []
    for key, value in metric_results.items():
        metric_names.append(key)
        metrics_vals.append(value)
        print(f"    {key:15s}: {value}")

    output = dict(zip(metric_names, metrics_vals))

    if config.metrics_calculator.save_results_path is not None:
        # you can use safetensors or other lib here
        save_path = ROOT_PATH / config.metrics_calculator.save_results_path
        (save_path).mkdir(exist_ok=True, parents=True)
        torch.save(output, save_path /"output_results.json")


def load_audio(file_path):
    """Load an audio file and return waveform and sample rate."""
    audio_tensor, sr = torchaudio.load(file_path, )
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    target_sr = 8000
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor

def calculate_metrics(metrics, evaluation_metrics, metric_args):
    for met in metrics["inference"]:
        evaluation_metrics.update(met.name, met(**metric_args))

def process_audio_files(target_path, preds_path, metrics, evaluation_metrics):
    """
    Process all audio files in target and prediction directories and calculate metrics.
    
    Args:
        target_path (Path): Path to target audio directory.
        preds_path (Path): Path to predictions audio directory.
        metrics (list): List of metric functions.
    
    Returns:
        dict: Metrics for all files.
    """
    mix_files = list((target_path / "mix").glob("*.wav"))
    for mix_file in tqdm(mix_files, desc="Processing files"):
        file_name = mix_file.name                                   # metric calcs only exist files
        if not os.path.exists((preds_path / "s1" / file_name)):
            continue

        mix = load_audio(target_path / "mix" / file_name)
        s1_target = load_audio(target_path / "s1" / file_name)
        s2_target = load_audio(target_path / "s2" / file_name)

        s1_pred = load_audio(preds_path / "s1" / file_name)
        s2_pred = load_audio(preds_path / "s2" / file_name)

        metric_args = {"output_audios": torch.stack([s1_pred, s2_pred]), 
                 "s1": s1_target,
                 "s2": s2_target,
                 "mix": mix}
        
        calculate_metrics(metrics, evaluation_metrics, metric_args)
        
    return evaluation_metrics.result()
    
if __name__ == "__main__":
    main()