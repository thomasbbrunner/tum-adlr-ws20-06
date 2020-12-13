# ADLR Project

- run build.sh to install requirements

## TODOs:
- [x] implement cVAE model
- [x] implement planar robot simulation with 2 and 3 DoF
- [x] implement paper's robot simulation
- [ ] implement rejection sampling for robotsim for comparing with cVAE
- [ ] implement random search for hyperparameter optimization (https://docs.ray.io/en/master/tune/)
- [ ] implement measure of how well model works (area where 95 percentile of samples lie)
- [ ] evaluate incorporating atan2 into the network (instead of tanh)
- [x] implement basic INN model
- [ ] implement jacobians of INN model
- [x] debug INN model
- [x] implement predict method of INN model
- [x] implement backward training of INN model
- [x] debug MMD loss
- [x] implement PCA
- [ ] debug padding
- [ ] implement abstract model class