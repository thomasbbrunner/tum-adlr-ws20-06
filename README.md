# ADLR Project

- run build.sh to install requirements

- Results of both models with different DoFs are stored in src/figures/evaluation/results/

## Hyper-parameters to tune

### cVAE

- learning rate (<= 0.01)
- batch_size
- amount of hidden layers per encoder/decoder  (>= 1)
- amount of neurons per hidden layer (>= 100)
- variational beta
- (loss weightings)

### INN

- learning rate (<= 0.001)
- batch_size
- amount of coupling layers (>= 2)
- amount of hidden layers per subnetwork (>= 1)
- amount of neurons per hidden layer (>= 100)
- (loss weightings)

## TODOs:
- [x] implement cVAE model
- [x] implement planar robot simulation with 2 and 3 DoF
- [x] implement paper's robot simulation
- [x] implement rejection sampling
- [ ] implement random search for hyperparameter optimization (https://docs.ray.io/en/master/tune/)
- [x] implement basic INN model
- [x] implement backward training of INN model
- [x] debug MMD loss