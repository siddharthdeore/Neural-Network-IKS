# Neural-Network-IKS
Deep Sequential Neural Network based Inverse kinematics solver for 2 degree of freedom planar robotic manipulator. 
![](https://img.shields.io/github/stars/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/forks/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/tag/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/release/siddharthdeore/Neural-Network-IKS.svg) ![](https://img.shields.io/github/issues/siddharthdeore/Neural-Network-IKS.svg)

### Problem Definition

For any input output system of with input state **X** and output states **Y** are related with transformation matrix **R** such a way that:

***Y=RX*** 

transformation matrix **R** maps input vector to output vector also known as forward kinematics.



For known output Y computing the required input state is done by method called Inverse kinematics algebraically expressed as:

**X=Y/R = R^-1\*Y** 
Generalized dimension of input is row vector of size (nx1), dimension of output state row vector is (mx1) and transformation matrix **R** is of dimension (mxn). In practice it is not guaranteed that **R** is invertible matrix, most often it is rectangular and hence we need to take pseudoinverse to get **X** from given **Y** .

### Configuration 
### Forward Kinematics 
### Inverse Kinematics
### Neural Network
### Installation
Create conda envirnoment

```conda create -n tensorflow_cpu pip python=3.6```

Activate environoment

```activate tensorflow_cpu```

```conda install numpy```

```conda install matplotlib```

```conda install tensorflow```

```conda install scikit-learn```

```conda install keras```

```conda install -c anaconda h5py ```

```conda install -c anaconda graphviz```

```conda install -c pydot```

### Usage
Run train_model.py to train model from given dataset
```python train_model.py```
Load and Test allready trained models
```python test_model.py```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
## License
[MIT](https://choosealicense.com/licenses/mit/)
