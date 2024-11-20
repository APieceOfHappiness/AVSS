from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
import torch

class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        all_losses = self.criterion(**batch)        
        
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())


        for met in metric_funcs:
            metrics.update(met.name, met(**batch))      
        
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        def peak_norm(pred_audio: torch.Tensor, ref_audio: torch.Tensor) -> torch.Tensor:
            peak_pred = torch.max(torch.abs(pred_audio))
            peak_ref = torch.max(torch.abs(ref_audio))
            normalized_audio = pred_audio * (peak_ref / peak_pred)
            return normalized_audio

        # method to log data from you batch
        # such as audio, text or images, for example

        if mode == "train":  # the method is called only every self.log_step steps
            pass
            # Log Stuff
            # self.writer.add_audio('output_audios1', 
            #                       peak_norm(batch['output_audios'][0][0], batch['mix'][0]),
            #                       sample_rate=8000)
            # self.writer.add_audio('output_audios2', 
            #                       peak_norm(batch['output_audios'][1][0], batch['mix'][0]), 
            #                       sample_rate=8000)
        else:
            # Log Stuff
            self.writer.add_audio('mix', batch['mix'][0], sample_rate=8000)
            self.writer.add_audio('s1', batch['s1'][0], sample_rate=8000)
            # self.writer.add_audio('s2', batch['s2'][0], sample_rate=8000)
            self.writer.add_audio('output_audios1', 
                                  peak_norm(batch['output_audio'][0], batch['mix'][0]),
                                  sample_rate=8000)
            # self.writer.add_audio('output_audios2', 
            #                       peak_norm(batch['output_audios'][1][0], batch['mix'][0]), 
            #                       sample_rate=8000)
