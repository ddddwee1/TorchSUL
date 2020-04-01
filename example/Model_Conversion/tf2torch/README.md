This folder shows example of transferring weight from tf_sul to torch_sul. 

You should change from python list (in tf_sul) to torch.nn.ModuleList (in torch_sul), and corresponding activation functions.

The computation flow should be consistent, and the script will dump weights layer-by-layer. 

1. Run tst.py in tf folder

2. Copy the dumped pkl file to torch folder, then run tst.py in torch folder. 

3. Then you can use torch model like eval.py

