import os 
import re
import glob
import shutil 

from typing import Union, Iterator
from loguru import logger 


# path utils 
class Path():
    def __init__(self, path: str):
        self.path = path 

    def __str__(self) -> str:
        return self.path

    @staticmethod
    def _tostr(obj: Union[str, 'Path']) -> str:
        assert isinstance(obj, (str, Path))
        if isinstance(obj, str):
            return obj 
        else:
            return obj.path 

    def __add__(self, other: Union[str, 'Path']) -> 'Path':
        path_new = os.path.join(self.path, self._tostr(other))
        return Path(path_new)
    
    def __radd__(self, other: Union[str, 'Path']) -> 'Path':
        path_new = os.path.join(self._tostr(other), self.path)
        return Path(path_new)
    
    def __contains__(self, s: str):
        return s in self.path

    def find_one(self, pattern: str) -> 'Path':
        new_path = None 
        for item in glob.glob(os.path.join(self.path, '*')):
            result = re.search(pattern, item.split('/')[-1])
            if result is not None:
                new_path = item 
        if new_path is not None:
            return Path(new_path)
        else:
            logger.warning(f'Cannot find pattern {pattern} in folder {self.path}')
            raise FileNotFoundError(f'Cannot find pattern {pattern} in folder {self.path}')
        
    def find(self, pattern: str) -> Iterator['Path']:
        for item in glob.glob(os.path.join(self.path, '*')):
            result = re.search(pattern, item.split('/')[-1])
            if result is not None:
                yield Path(item)

    def auto(self) -> 'Path':
        sub_paths = glob.glob(os.path.join(self.path, '*'))
        if len(sub_paths)==0:
            raise FileNotFoundError(f'Auto: No file under {self.path}')
        else:
            return Path(sub_paths[0])
        
    def copy_to(self, target: 'Path'):
        os.makedirs(os.path.dirname(target.path), exist_ok=True)
        shutil.copy(self.path, target.path)
    
    def copy_dir_to(self, target: 'Path'):
        os.makedirs(os.path.dirname(target.path), exist_ok=True)
        if os.path.exists(target.path):
            shutil.rmtree(target.path)
        shutil.copytree(self.path, target.path)

    def remove(self):
        os.remove(self.path)
    
    def remove_dir(self):
        shutil.rmtree(self.path)

    def makedirs(self):
        os.makedirs(self.path, exist_ok=True)
    
    def replace(self, src: str, tgt: str) -> 'Path':
        path_new = self.path.replace(src, tgt)
        return Path(path_new)
