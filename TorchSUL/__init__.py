from .Utils import Config
from . import Base, Model, Quant, Tools

sul_tool = Tools

# modify loguru format. Add \n in front, in case other modules overrides stderr
import sys 
from loguru import logger 


class PrintStream():
    def write(self, message):
        print(message, end='')


logger.remove()
fmt = '<g>{time:YYYY-MM-DD HH:mm:ss}</g> | PID: <c>{process}</c> | <lvl>{level}</lvl> - <lvl>{message}</lvl> '
stream_obj = PrintStream()
# logger.add(sys.stderr, format=fmt)
logger.add(stream_obj, format=fmt, colorize=True)


