param.set <- makeParamSet(
  # use a subset of 978 landmark features only to speed up training
  makeDiscreteParam("use_landmark_genes", values=c(True)),

  # large batch_size only makes sense when warmup_lr is on
  makeDiscreteParam("batch_size", values=c(32, 64, 128, 256, 512, 1024)),

  # use consecutive 1000-neuron layers to facilitate residual connections
  makeDiscreteParam("dense",
	            values=c("1000",
                             "1000 1000",
                             "1000 1000 1000",
                             "1000 1000 1000 1000",
                             "1000 1000 1000 1000 1000")),

  makeDiscreteParam("dense_feature_layers",
	            values=c("1000",
                             "1000 1000",
                             "1000 1000 1000",
                             "1000 1000 1000 1000",
                             "1000 1000 1000 1000 1000")),

  makeDiscreteParam("residual", values=c(True, False)),

  makeDiscreteParam("activation", values=c("relu", "sigmoid", "tanh")),

  makeDiscreteParam("optimizer", values=c("adam", "sgd", "rmsprop")),

  makeNumericParam("learning_rate", lower=0.00001, upper=0.1),

  makeDiscreteParam("reduce_lr", values=c(True, False)),

  makeDiscreteParam("warmup_lr", values=c(True, False)),

  makeIntegerParam("epochs", lower=100, upper=200),
)
