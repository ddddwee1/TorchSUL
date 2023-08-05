from setuptools import setup, find_packages 

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name='TorchSUL',
    version='0.2.4',
    description='Simple but useful layers for Pytorch',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Cheng Yu',
    author_email='chengyu996@gmail.com',
    url='https://github.com/ddddwee1/TorchSUL',
    install_requires = [
        'tqdm',
        'opencv-python',
        'matplotlib',
        'pyyaml',
        'moviepy',
    ]
)

install_requires = []

if __name__ == '__main__':
	setup(**setup_args)
