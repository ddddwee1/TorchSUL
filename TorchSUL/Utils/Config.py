from typing import Any

import yaml


class ConfigDict():
	'''
	Easy-to-use dict
	Python dictionary: d[A][B]  ->  ConfigDict:  d.A.B
	'''
	def __init__(self, d: dict):
		self.dict_str = str(d)
		self.dict = self.parse_dict(d)

	@staticmethod
	def parse_dict(d):
		if isinstance(d, dict):
			res = {}
			for k in d:
				if isinstance(d[k], dict):
					buff = ConfigDict(d[k])
				else:
					buff = ConfigDict.parse_dict(d[k])
				res[k] = buff
		elif isinstance(d, list):
			res = [ConfigDict.parse_dict(i) for i in d]
		else:
			res = d
		return res 

	def __getattr__(self, key) -> Any | "ConfigDict":
		if not key[0]=='_':
			return self.dict[key]
		else:
			super().__getattr__(key)             # type: ignore

	def __str__(self) -> str:
		return self.dict_str


def load_yaml(f: str) -> ConfigDict:
	with open(f) as file:
		config: dict = yaml.safe_load(file)
	return ConfigDict(config) 


if __name__=='__main__':
	# for debug purpose
	d = {'AB': 1, 'BC':[2,3], 'CD': {'AAA':1, 'BBB':2}}
	dd = ConfigDict(d)
	print(dd.CD.AAA)

	d2 = load_yaml('../abc.yaml')
	print(d2.MODEL.AAB)
