# -*- coding: utf-8 -*-
"""
:Author: Jaekyoung Kim
:Date: 2017. 11. 13.
"""
import logging as lg

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0


def get_logger(logger_name, log_dir, file_name, level):
    logger = lg.getLogger(logger_name)
    fomatter = lg.Formatter('[%(levelname)s|%(filename)s:%(lineno)s] %(asctime)s > %(message)s')

    # 스트림과 파일로 로그를 출력하는 핸들러를 각각 만든다.
    file_handler = lg.FileHandler(log_dir + '/' + file_name + '.log')
    stream_handler = lg.StreamHandler()

    # 각 핸들러에 포매터를 지정한다.
    file_handler.setFormatter(fomatter)
    stream_handler.setFormatter(fomatter)

    # 로거 인스턴스에 스트림 핸들러와 파일핸들러를 붙인다.
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(level)

    return logger
