# ECE452 Project \#1

To get started, install the following python dependencies:

- PyTorch
- Matplotlib
- Seaborn
- ScikitLearn

probably best to do this in a venv

-----

Then to train a single network with some parameters, run 

```
python3 Train.py --epochs <num epochs> --lr <learning rate> --plot <plot name> --opt <optimizer> --loss <loss function>
``` 

Run `python3 Train.py -h` for all options

-----

To train a whole lot of networks to find best performance, tweak `TrainAll.py` as needed and run `python3 TrainAll.py`

-----

To test a network, run 

```
python3 Test.py <path to .pt file>
```

**NOTE:** The file must contain the structure of the network too, not just the state\_dict.

-----

To test all the generated networks, run

```
python3 TestAll.py
```

This will run all the tests, and generate coefficient matrices.
