from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from transformers.utils import logging

logger = logging.get_logger('transformers')


class TrainerLoggerCallback(TrainerCallback):

    def on_log(self, args, state, control, logs=None, **kwargs):
        # A bare [`TrainerCallback`] that just prints the logs.
        _ = logs.pop("total_flos", None)
        # if state.is_local_process_zero:
        #     logger.info(logs)


class CrmmTrainerCallback(TrainerCallback):

    def __init__(self, runner) -> None:
        self.runner = runner

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        super().on_epoch_end(args, state, control, **kwargs)
        self.runner.predict_on_test()
