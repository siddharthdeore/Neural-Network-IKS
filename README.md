# Neural-Network-IKS
Deep Sequential Neural Network based Inverse kinematics solver for 2 degree of freedom planar robotic manipulator. 
![](https://img.shields.io/github/stars/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/forks/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/tag/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/issues/siddharthdeore/Neural-Network-IKS.svg)

### Problem Definition

I'm looking forward to solve over deterimined system with input state **X** and output states **Y** are related with transformation matrix **A** such a way that:

<img src="https://latex.codecogs.com/svg.latex?Y&space;=&space;A&space;X">


transformation matrix **A** maps input vector to output vector also known as forward kinematics.

For known output Y, computing the required input state is done by method called Inverse kinematics algebraically expressed as:

<img src="https://latex.codecogs.com/svg.latex?X&space;=&space;A^{-1}&space;Y&space;=&space;A^T(AA^T)^{-1}&space;Y">

Generalized dimension of input is row vector of size (nx1), dimension of output state row vector is (mx1) and transformation matrix **A** is of dimension (mxn). In practice it is not guaranteed that **A** is invertible matrix, most often it is rectangular and hence we need to take pseudoinverse to get **X** from given **Y** .

### Neural Network
Fully connected Feed Forward

#### Training results

Accuracy
![acc](/fig/acc.png)

Loss
![loss](/fig/loss.png)

### Installation
Create conda envirnoment

```sh
conda create -n tensorflow pip python=3.6
```

Activate environoment

```sh
activate tensorflow
```

Install required pacages

```sh
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorflow 
conda install -c conda-forge keras 
conda install -c anaconda h5py
```

or for GPU

```sh
conda install -c anaconda tensorflow-gpu 
conda install -c anaconda keras-gpu 
```

Read more about tensorflow installation instruction instructions [tensorflow](https://www.tensorflow.org/install)
and Keras installation instructions [keras.io](https://keras.io/#installation)

---
### Usage
Run train_model.py to train model from given dataset

```python train_model.py```

Load and Test allready trained models

```python test_model.py```

---
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Cite this repository in publication
```
@misc{Neural-Network-IKS,
  author = {Deore, Siddharth},
  title = {Neural Network based Inverse Kinematics of planar robotic arm},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/siddharthdeore/Neural-Network-IKS}},
}
```
---
## License
[MIT](https://choosealicense.com/licenses/mit/)
