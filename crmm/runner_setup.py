import json
import logging as syslogging
import os
import random
import string
from datetime import datetime

from transformers.utils import logging

from .arguments import CrmmTrainingArguments
from .utils import utils
from .utils.log_handler import ColorFormatter

logger = logging.get_logger('transformers')


def setup(exp_args: CrmmTrainingArguments, data_args=None, model_args=None):
    if exp_args.auto_create_model_dir:
        initial_timestamp = datetime.now()
        root_dir = exp_args.root_dir
        if not os.path.isdir(root_dir):
            raise IOError(
                "Root directory '{}', where the directory of the experiment will be created, must exist".format(
                    root_dir))

        output_dir = os.path.join(root_dir, exp_args.task)

        formatted_timestamp = initial_timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        rand_suffix = "".join(random.choices(string.ascii_letters + string.digits, k=3))
        output_dir += "_" + formatted_timestamp + "_" + rand_suffix
        exp_args.output_dir = os.path.join(output_dir, 'output')
        exp_args.logging_dir = os.path.join(output_dir, 'logging')
        utils.create_dirs([exp_args.output_dir, exp_args.logging_dir])

    color_fmt = ColorFormatter()
    file_handler = syslogging.FileHandler(os.path.join(exp_args.output_dir, 'output.log'), )
    file_handler.setFormatter(color_fmt)

    console_handler = syslogging.StreamHandler()
    console_handler.setFormatter(color_fmt)

    # note:  disable_default_handler!
    logging.disable_default_handler()
    logging.set_verbosity_debug()
    logging.add_handler(file_handler)
    logging.add_handler(console_handler)

    # Save configuration as a (pretty) json file
    with open(os.path.join(exp_args.output_dir, 'training_arguments.json'), 'w') as f:
        json.dump(exp_args.to_sanitized_dict(), f, indent=4, sort_keys=True)

    # with open(os.path.join(exp_args.output_dir, 'model_arguments.json'), 'w') as f:
    #     json.dump(model_args., f, indent=4, sort_keys=True)
    #
    # with open(os.path.join(exp_args.output_dir, 'data_arguments.json'), 'w') as f:
    #     json.dump(data_args.to_sanitized_dict(), f, indent=4, sort_keys=True)

    logger.info("Stored configuration file in '{}'".format(exp_args.output_dir))

    return exp_args
