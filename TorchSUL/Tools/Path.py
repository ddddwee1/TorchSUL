import glob
import os
import re
import shutil
from typing import Iterator, Union

from loguru import logger


# path utils 
class Path(os.PathLike[str]):
    def __init__(self, path: str):
        self.path = path 

    def __str__(self) -> str:
        return self.path
    
    def __fspath__(self) -> str:
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
    
    def __getitem__(self, idx: int) -> str:
        return self.path.split('/')[idx]

    def find_one(self, pattern: str, use_regex: bool = False) -> 'Path':
        if not use_regex:
            pattern = '^' + pattern.replace('*', '.*') + '$'
        new_path = None 
        for item in glob.glob(os.path.join(self.path, '*')):
            result = re.search(pattern, item.split('/')[-1])
            if result is not None:
                new_path = item 
        if new_path is not None:
            return Path(new_path)
        else:
            raise FileNotFoundError(f'Cannot find pattern {pattern} in folder {self.path}')
        
    def find(self, pattern: str, use_regex: bool = False) -> Iterator['Path']:
        if not use_regex:
            pattern = '^' + pattern.replace('*', '.*') + '$'
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
            target.remove_dir()
        shutil.copytree(self.path, target.path)

    def remove(self):
        try:
            os.remove(self.path)
        except Exception as e:
            logger.warning(f'Cannot remove {self.path}. {e}')
    
    def remove_dir(self):
        try:
            shutil.rmtree(self.path)
        except Exception as e:
            logger.warning(f'Cannot remove {self.path}. {e}')

    def makedirs(self):
        os.makedirs(self.path, exist_ok=True)
    
    def replace(self, src: str, tgt: str) -> 'Path':
        path_new = self.path.replace(src, tgt)
        return Path(path_new)

    def exists(self) -> bool:
        return os.path.exists(self.path)
    
    def basename(self) -> str:
        return os.path.basename(self.path)
    
    def dirname(self) -> 'Path':
        return Path(os.path.dirname(self.path))

