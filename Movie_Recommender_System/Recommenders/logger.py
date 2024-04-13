import logging, os
from datetime import date
from jsonformatter import JsonFormatter

log_format = """{
        "Log_Time": "%(asctime)s", 
        "Levelname":  "%(levelname)s",
        "Filename": "%(filename)s", 
        "Module": "%(module)s",
        "Lineno": "%(lineno)d",
        "FuncName": "%(name)s",
        "Message": "%(message)s"
    }"""
year = date.today().strftime("%Y")
month = date.today().strftime("%B")[:3]


def file_logger(
    file_function: str, logger_task: str, folder: str, string_format: str = log_format
) -> logging.Logger:
    """
    Creates logger object that sends message to destination file handler in specified format
    in a specified destination folder.
    :param file_function: File name indicating what kind of functional messages
    :param logger_task: What specific task logger will perform?
    :param folder: Folder where logger file will be created
    :param string_format: Logging message format in json
    :return: logging.Logger (logger object)
    """
    global year, month
    if not os.path.exists(f"Logs/{folder}"):
        os.makedirs(f"Logs/{folder}")
    logger = logging.getLogger(logger_task)
    logger.setLevel(logging.DEBUG)
    filehandler = logging.FileHandler(
        f"Logs/{folder}/{file_function}_{month}_{year}.log"
    )
    formatter = JsonFormatter(
        string_format, ensure_ascii=False, mix_extra=True, mix_extra_position="tail"
    )
    filehandler.setFormatter(formatter)
    logger.addHandler(filehandler)
    return logger


def add_handler(
    logger: logging.Logger, file_function: str, folder: str
) -> logging.FileHandler:
    """
    Add file handler to existent logger object.
    :param logger: logger object existent currently
    :param file_function:
    :param folder: Folder where logger file will be created
    :return: logging.FileHandler
    """
    global year, month
    filehandler = logging.FileHandler(
        f"Logs/{folder}/{file_function}_{month}_{year}.log"
    )
    logger.addHandler(filehandler)
