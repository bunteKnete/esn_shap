
predict_ml_model_esn <- function(Win,
                                 W,
                                 Wout,
                                 x,
                                 input_data,
                                 leak,
                                 pred_idx,
                                 out_size = 1){
  
  # Run the trained ESN in a generative mode. No need to initialize here, 
  # because x is initialized with training data and we continue from there.
  
  pred_len <- length(pred_idx)
  Y <- matrix(0, out_size, pred_len)
  
  in_size <- dim(input_data)[2]
  
  for (t_idx in pred_idx){
    u <- matrix(input_data[t_idx,], in_size)
    x <- (1 - leak) * x + leak * tanh( Win %*% rbind(1, u) + W %*% x )
    y <- Wout %*% rbind(1, u, x)
    Y[, t_idx - pred_idx[1] + 1] <- y
  }
  
  return(Y)
}
#-------------------------------------------------------------------------------

predict_esn_shap <- function(model, input_data){
  # Wrapper for the prediction function to work with the syntax of 
  # kernelshap.
  
  Win  <- model[[1]]
  W    <- model[[2]]
  Wout <- model[[3]]
  x    <- model[[4]]
  leak <- model[[5]]
  
  num_records <- dim(input_data)[1]
  test_idx    <- 1:num_records
  
  Y <- predict_ml_model_esn(Win, 
                            W, 
                            Wout, 
                            x, 
                            input_data, 
                            leak, 
                            test_idx, 
                            out_size = 1
  )
  
  return(Y)
}
