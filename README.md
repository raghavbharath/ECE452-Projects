# ECE 452 Project 1 - A customized MLP for digit recognition

This project is an MLP-based handwritten digit recognition across 3 parts. I completed it with my partners, Arnav Revankar and Daniel Assaf. 

Part 1 uses PyTorch for training and a custom NumPy inference engine for testing. Parts 2 and 3 use MATLAB's Neural Network Toolbox.

## Setup

To test out the project, you can install the following python dependencies in a venv:

* PyTorch
* Matplotlib
* ScikitLearn
* Seaborn

Under "Project1/Part1/" I have a bare-bones requirements.txt you can import into the venv to try it out.

## Part 1 (Python)

To train a single network with some parameters, you can run:

```
python3 Train.py --epochs <num epochs> --lr <learning rate> --plot <plot name> --opt <optimizer> --loss <loss function>
```

Run `python3 Train.py -h` for all options.

To train a whole lot of networks to find best performance, tweak `TrainAll.py` as needed and run:

```
python3 TrainAll.py
```

To test a network, run:

```
python3 Test.py <path to .pt file>
```

NOTE: The file must contain the structure of the network too, not just the state_dict.

To test all the generated networks, run:

```
python3 TestAll.py
```

This will run all the tests and generate confusion matrices.

## Parts 2 & 3 (MATLAB)

Open the `.m` scripts in MATLAB:

* `P1_train_export_IDX.m` to train the network and export weights
* `P1_custom_inference_STUDENT.m` to run custom inference without built-in prediction functions

See the reports folder for detailed results and analysis.
