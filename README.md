<h1 align="center">Audio-Visual Source Separation (AVSS)</h1>

## About

The model based on [Dual-path RNN](https://arxiv.org/abs/1910.06379).

<!-- See [wandb report](https://wandb.ai/dungeon_as_fate/pytorch_template_asr_example). -->

## Results on train and validation:

| Model  | SI_SNRi | SDRi  | PESQ | STOI |
|--------|---------|-------|------|------|
| DPRNN  | 13.01   | 12.67 | 2.9  | 0.9  |

## Installation

0. Create new conda environment:
```bash
conda create -n avss_env python=3.10

conda activate avss_env
``` 

1. Install all requirements.
```bash
pip install -r requirements.txt
```

## Inference
To save the separation predictions made by the pre-trained model (and enabling immediate calculation of metrics), use the following script:
    ```bash
    python inference.py \
           datasets.dataset_path=<dir of custom dataset> \
           datasets.audio_ref=<True or False> \
           inferencer.calc_metrics=<True or False> \
           inferencer.device=<device> \
           inferencer.save_path=<name of the folder for predicted separation> \
           inferencer.from_pretrained="./saved/dprnn_pretrained/checkpoint-epoch58.pth"
    ```

   (datasets.audio_ref is True if s1 and s2 are in the dataset_path, for inferencer.calc_metrics=True, datasets.audio_ref must be True)

   To calculate metrics (SI-SNR, SDRi, PESQ, and STOI) for the given directories of target and predicted paths, use the following script:
   ```bash
    python calc_metrics.py \
           metrics_calculator.target_path=<dir of custom dataset> \
           metrics_calculator.preds_path=<dir of custom dataset> \
           metrics_calculator.save_results_path=<dir of folder to save json with results>
   ```

## Train
   Training script for DPRNN:
   ```bash
   python train.py \
          trainer.device=<device> trainer.override=True \
          writer.log_checkpoints=True \
          writer.run_name=dprnn_train
   ```
