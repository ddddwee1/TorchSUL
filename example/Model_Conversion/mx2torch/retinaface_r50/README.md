0. Get models from [here](https://www.dropbox.com/s/cgxbvnwexlao1qq/retinaface_models.zip?dl=0)

1. Run 

```
python res50_transfer_weight.py

cd ./rcnn/cython
python setup.py build_ext --inplace
cd ../../
```

2. Finished

Just use 

```
test.py
```
