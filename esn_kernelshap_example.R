# Working example for kernelshap with an echo state network.

library(kernelshap)

source("esn_predict.R")

# Load the trained model
load("esn_model.Rdata")

# Load the training (i.e. input) data.
# The data are normalized temperature data for mortality prediction. From this
# gridded data a single value for the mortality is predicted. All 2014 grid
# points are supposed to be the features for which the importances shall be
# calculated.
load("train_data.Rdata")

# Background temperatures. Use full training data set since it's a very
# small dataset.
bg_temp   <- train_data[seq(1, 60, 10),]

# Test data on which SHAP values are computed.
test_data <- train_data[54:60,]

ks <- kernelshap(object   = esn_model_shap,
                 X        = test_data,
                 bg_X     = bg_temp,
                 pred_fun = predict_esn_shap
)

save(ks, file = "./shap_vals.Rdata")
