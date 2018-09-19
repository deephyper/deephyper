# see https://cran.r-project.org/web/packages/ParamHelpers/ParamHelpers.pdfmakeNum
# the parameter names should match names of the arguments expected by the benchmark

param.set <- makeParamSet(
  makeDiscreteParam("batch_size", values = c(32, 64, 128)),
  makeDiscreteParam("dense", values = c("500 100 50", "1000 500 100 50", "2000 1000 500 100 50")),
  makeDiscreteParam("activation", values = c("relu", "sigmoid", "tanh")),
  makeDiscreteParam("optimizer", values = c("adam", "sgd", "rmsprop")),
  makeNumericParam("drop", lower = 0, upper = 0.5),
  makeDiscreteParam("conv", values = c("0 0 0", "5 5 1", "10 10 1 5 5 1")),

  ## DEBUG PARAMETERS: DON'T USE THESE IN PRODUCTION RUN
#  makeIntegerParam("feature_subsample", lower=500, upper=500),
#  makeIntegerParam("train_steps", lower=100, upper=100),
#  makeIntegerParam("val_steps", lower=10, upper=10),
#  makeIntegerParam("test_steps", lower=10, upper=10),
#  makeIntegerParam("epochs", lower = 3, upper = 3)
  ## END DEBUG PARAMS
)
