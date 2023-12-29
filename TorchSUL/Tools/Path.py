from __future__ import annotations

import glob
import os


# path utils 
class Path():
    path: str
    
    def __init__(self, path: str) -> None:
        self.path = path 

    def __add__(self, p2: Path | str) -> Path:
        if isinstance(p2, str):
            p2 = os.path.join(self.path, p2)
            return Path(p2)
        elif isinstance(p2, Path):
            p2 = os.path.join(self.path, p2.path)
            return Path(p2)

    def auto(self) -> Path:
        p2 = glob.glob(os.path.join(self.path, '*'))[0]
        return Path(p2)

    def find(self, prefix: str) -> Path:
        files = glob.glob(os.path.join(self.path, '*'))
        for f in files:
            if prefix in f.split('/')[-1]:
                return Path(f)
        raise FileNotFoundError(f'Cannot find [{prefix}] under path [{self.path}]')

    def __repr__(self) -> str:
        return self.path 

    def tostr(self) -> str:
        return self.path 

    def __str__(self) -> str:
        return self.path

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def makedirs(self):
        return os.makedirs(self.path, exist_ok=True)
        
