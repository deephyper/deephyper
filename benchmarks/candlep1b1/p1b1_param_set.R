# see https://cran.r-project.org/web/packages/ParamHelpers/ParamHelpers.pdfmakeNum
# the parameter names should match names of the arguments expected by the benchmark

# Current best val_corr: 0.96 for ae, 0.86 for vae
# We are more interested in vae results

param.set <- makeParamSet(
  # we optimize for ae and vae separately
  makeDiscreteParam("model", values=c("ae", "vae")),

  # latent_dim impacts ae more than vae
  makeDiscreteParam("latent_dim", values=c(2, 8, 32, 128, 512)),

  # use a subset of 978 landmark features only to speed up training
  makeDiscreteParam("use_landmark_genes", values=c(True)),

  # large batch_size only makes sense when warmup_lr is on
  makeDiscreteParam("batch_size", values=c(32, 64, 128, 256, 512, 1024)),

  # use consecutive 978-neuron layers to facilitate residual connections
  makeDiscreteParam("dense", values=c("2000 600",
                                      "978 978",
				      "978 978 978",
				      "978 978 978 978",
				      "978 978 978 978 978",
				      "978 978 978 978 978 978")),

  makeDiscreteParam("residual", values=c(True, False)),

  makeDiscreteParam("activation", values=c("relu", "sigmoid", "tanh")),

  makeDiscreteParam("optimizer", values=c("adam", "sgd", "rmsprop")),

  makeNumericParam("learning_rate", lower=0.00001, upper=0.1),

  makeDiscreteParam("reduce_lr", values=c(True, False)),

  makeDiscreteParam("warmup_lr", values=c(True, False)),

  makeNumericParam("drop", lower=0, upper=0.9),

  makeIntegerParam("epochs", lower=100, upper=200),
)
