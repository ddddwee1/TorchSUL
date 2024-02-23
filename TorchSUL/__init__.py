from .Utils import Config
from . import Base, Model, Quant, Tools

sul_tool = Tools

# modify loguru format. Add \n in front, in case other modules overrides stderr
import sys 
from loguru import logger 

logger.remove()
fmt = '<g>{time:YYYY-MM-DD HH:mm:ss}</g> | PID: <c>{process}</c> | <lvl>{level}</lvl> - <lvl>{message}</lvl> '
logger.add(sys.stderr, format=fmt)


