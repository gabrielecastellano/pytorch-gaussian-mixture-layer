## Pytorch Gaussian Mixture Layer

This repository implements Gaussian Mixture Layer in pytorch. The layer extends nn.Module and can therefore be used
 within any sequence of modules in place of any other pytorch layer (e.g, in place of the linear classifier).
 The GMM Layer can be trained with SDG or any other method that leverages the autograd feature provided by pytorch.
 Please find an example of usage within a CNN architecture in the file [example.py](example.py) (see below).

### Module Architecture

The GMM layer behaves as a classifier, assuming N classes and that each class is modeled by G different gaussian 
 components. The module assumes a flat input and produces N log probabilities as output. Each log probability
 represents the likelihood that the input belongs to a certain class. Since that multiple mixture are used to model the 
 same class, the retrieved likelihood is computed as the sum of the probabilities of all the G components for that 
 class.

#### Main Arguments

- **Raw input dimensions:** number of dimensions of the raw data provided as input. Could be larger than the number of 
  dimensions of the normal distribution (see below).
- **Reduced dimension D:** the number of dimensions actually handled by the gaussian mixture. If different from the size 
  of the raw input, this pre-processed to apply dimensionality reduction through random-projection.
- **Classes N:** number of labels featured by the data.
- **Per-class components G:** the number of gaussian distributions used to model a single class. The total number of 
  component for the gaussian mixture will be N*G. 

You can easily instantiate a gmm layer as follows:

```python
> from gmml import GMML
> gmm_layer = GMML(100, 100, 10, n_component=1, cov_type="tril")
```

This will create a gmm layer that accepts 100-dimensional vectors as input and fits them using a mixture of 10 gaussian,
 one for each class. The _cov_type_ argument specifies the type of covariance matrix to be used (diag|tril|full). The 
 diagonal option produces a lighter model that is easier to learn, but provides less flexibility to fit more complex 
 data.

#### Parameters

For every component k, the module features the following set of parameters:
 
- **Mean:** D-dimensional vector, represents the mean value for component k;
- **Sigma:** D x D matrix, used to sample the covariance matrix of component k;
- **Omega:** scalar value, used to sample the weight of component k in the mixture.

#### Forwarding Process

The module operates on batch of data with of dynamic size. For simplicity, following description refers to the process 
 of a single input vector. The input is first forwarded to a linear layer that performs (if there is a mismatch between 
 the raw input dimensions and D), dimensionality reduction through random projection. This first transformation produces 
 a D-dimensional data. For each component of the mixture, we compute the log probability of the input data to belong to
 that component, based on the mean and covariance matrix of the normal distribution concerning that component. The 
 obtained probabilities for each component are then weighted according to the current mixture weights, thus obtaining 
 the log probability vector of the mixture distribution. Last, The probabilities of components that model the same class 
 are summed up using the logsumexp trick. The final output is therefore a vector with N log probabilities, one for each
 class.

#### Gaussian Mixture Constraints

The parameters of a GMM are supposed to satisfy some constraints:

- The mixture weights must sum up to 1;
- The Covariance matrix must be positive definite.

However, as these parameters are modified during the training procedure, we need to ensure that distributions used to 
 compute probability values are always compliant with the constraints above.

To comply with this requirement, we do not use the parameters above directly when building the new gaussian mixture. 
 Instead, we leave Omega and Sigma as free parameters, and we sample feasible covariance matrix and mixture weights from 
 them. This is done by invoking the function

```python
> gmm_layer.sample_parameters()
```

whenever the free parameters are updated (e.g., after invoking backpropagation).

## Basic GMM Layer example with MNIST

The file [example.py](example.py) provides an example of a simple CNN architecture that performs classification, where 
 a GMM layer is used as the last layer. You can run the example as follows:  

```bash
$ pip install -r requirements.txt
$ python example.py --batch-size 64 --epochs 20 --lr 1.0 --seed 1
```