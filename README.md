# DL-Custom-Model
A CNN model trained on the CIFAR-10 dataset from scratch.

# Installations:
Make sure that python3 and pip is installed. These libraries need to be installed.
```bash
pip install numpy tensorflow scikit-learn matplotlib
```
To verify the installations:
```bash
pip freeze | grep -E 'numpy|tensorflow|scikit-learn|matplotlib'
```

# How to Run:
Clone the repository and run the python code:
```bash
git clone https://github.com/sakhij/DL-Custom-Model.git
```
Use the following file to print values per epoch:
```bash
python3 CustomModelWithValues.py
```
Else use:
```bash
python3 CustomModelCifar10.py
```

# Expected Output:(Use as Reference)
The graphs for 20 epochs look like:
![Graphs for Cipher10](https://github.com/user-attachments/assets/c0eccc5b-dc37-4c2d-9ac2-f0a20e7184f6)
Model's metrics are:<br>
`Precision Metric: 0.7595`<br>
`Recall Metric: 0.7555`
