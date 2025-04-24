library(jsonlite)

print_model_summary <- function(file_path) {
  if (!file.exists(file_path)) {
    stop("File not found: ", file_path)
  }

  # Read JSON from file
  json_text <- readLines(file_path, warn = FALSE)
  json_combined <- paste(json_text, collapse = "\n")
  model <- fromJSON(json_combined)

  cat("====== PPO MODEL SUMMARY ======\n\n")

  # Episode
  cat("Episode:", model$IN_EPISODE, "\n\n")

  # Policy Network
  cat("---- POLICY NETWORK ----\n")
  cat("Layers:\n")
  cat(" - Layer 1 weights:", dim(model$POLICY1)[[1]], "x", dim(model$POLICY1)[[2]], "\n")
  cat(" - Layer 2 weights:", dim(model$POLICY2)[[1]], "x", dim(model$POLICY2)[[2]], "\n")
  cat(" - Layer 3 weights:", dim(model$POLICY3)[[1]], "x", dim(model$POLICY3)[[2]], "\n")
  cat(" - Output weights:", dim(model$POLICY_WEIGHTS_OUTPUT)[[1]], "x", dim(model$POLICY_WEIGHTS_OUTPUT)[[2]], "\n\n")

  # Value Network
  cat("---- VALUE NETWORK ----\n")
  cat("Layers:\n")
  cat(" - Layer 1 weights:", dim(model$VALUE1)[[1]], "x", dim(model$VALUE1)[[2]], "\n")
  cat(" - Layer 2 weights:", dim(model$VALUE2)[[1]], "x", dim(model$VALUE2)[[2]], "\n")
  cat(" - Layer 3 weights:", dim(model$VALUE3)[[1]], "x", dim(model$VALUE3)[[2]], "\n")
  cat(" - Output weights:", dim(model$VALUE_WEIGHTS_OUTPUT)[[1]], "x", dim(model$VALUE_WEIGHTS_OUTPUT)[[2]], "\n\n")

  # BatchNorm
  cat("---- BATCH NORM (POLICY) ----\n")
  for (i in 1:length(model$POLICY_BATCH_NORM_GAMMA)) {
    cat(sprintf(
      "Layer %d: Gamma[%.3f ...], Beta[%.3f ...]\n",
      i, model$POLICY_BATCH_NORM_GAMMA[[i]][1], model$POLICY_BATCH_NORM_BETA[[i]][1]
    ))
  }

  cat("\n---- BATCH NORM (VALUE) ----\n")
  for (i in 1:length(model$VALUE_BATCH_NORM_GAMMA)) {
    cat(sprintf(
      "Layer %d: Gamma[%.3f ...], Beta[%.3f ...]\n",
      i, model$VALUE_BATCH_NORM_GAMMA[[i]][1], model$VALUE_BATCH_NORM_BETA[[i]][1]
    ))
  }

  cat("\n==== END OF SUMMARY ====\n")
}
