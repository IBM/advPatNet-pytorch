import os
import sys
import datetime
import logging
import shutil


def date_uid():
    """Generate a unique id based on date.

    Returns:
        str: Return uid string, e.g. '20171122171307111552'.

    """
    return str(datetime.datetime.now()).replace('-', '') \
        .replace(' ', '').replace(':', '').replace('.', '')


def get_logger(checkpoint_path, filename, filemode='w'):
    """
    Get the root logger
    :param checkpoint_path: only specify this when the first time call it
    :return: the root logger
    """
    if filemode == 'w' and os.path.exists(os.path.join(checkpoint_path, filename)):
        print ("\n**************************************************", flush=True)
        print("Found old results, copying it to avoid for overwritten.", flush=True)
        target_path = checkpoint_path.rstrip('/')
        i = 0
        curr_target_path = target_path + f".{i}"
        while True:
            if os.path.exists(curr_target_path):
                i += 1
                curr_target_path = target_path + f".{i}"
            else:
                break
        print(f"Copying old log folder to {curr_target_path}", flush=True)
        print ("**************************************************\n", flush=True)
        shutil.copytree(checkpoint_path, curr_target_path)

    if checkpoint_path:
        logger = logging.getLogger()
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        stream_hdlr = logging.StreamHandler(sys.stdout)
#        log_filename = date_uid()
        file_hdlr = logging.FileHandler(os.path.join(checkpoint_path, filename), mode=filemode)
        stream_hdlr.setFormatter(formatter)
        file_hdlr.setFormatter(formatter)
        logger.addHandler(stream_hdlr)
        logger.addHandler(file_hdlr)
        logger.setLevel(logging.INFO)
    else:
        logger = logging.getLogger()
    return logger
