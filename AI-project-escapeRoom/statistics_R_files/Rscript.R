library(jsonlite)
library(ggplot2)
library(gridExtra)

# Function to flatten matrices and vectors to a single vector for each episode
flatten_weights_and_biases <- function(weights_list, biases_list) {
  # Flatten weights and biases into a list of vectors
  weights_flat <- lapply(weights_list, function(w) as.vector(w))
  biases_flat <- lapply(biases_list, function(b) as.vector(b))
  return(list(weights = weights_flat, biases = biases_flat))
}

# Function to print model summary from the JSON file
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

# Function to create and plot learning progress from the PPO model
plot_learning_progress <- function(file_path) {
  if (!file.exists(file_path)) {
    stop("File not found: ", file_path)
  }

  # Read JSON from file
  json_text <- readLines(file_path, warn = FALSE)
  json_combined <- paste(json_text, collapse = "\n")
  model_data <- fromJSON(json_combined)

  # Flatten actor and critic weights and biases
  actor_flat <- flatten_weights_and_biases(model_data$IN_ACTOR_WEIGHTS, model_data$IN_ACTOR_BIASES)
  critic_flat <- flatten_weights_and_biases(model_data$IN_CRITIC_WEIGHTS, model_data$IN_CRITIC_BIASES)

  # Check if the lengths match across all flattened parameters
  num_episodes <- length(actor_flat$weights)

  if (length(actor_flat$weights) == length(actor_flat$biases) &&
    length(critic_flat$weights) == length(critic_flat$biases)) {
    # Create episode numbers based on the number of episodes
    episode_number <- rep(1:num_episodes, each = length(actor_flat$weights[[1]])) # Replicate episode numbers

    # Create data frame for the plots
    model_df <- data.frame(
      Episode = rep(episode_number, each = length(actor_flat$weights[[1]])),
      ActorWeights = unlist(actor_flat$weights),
      ActorBiases = unlist(actor_flat$biases),
      CriticWeights = unlist(critic_flat$weights),
      CriticBiases = unlist(critic_flat$biases)
    )

    # Plot Actor Weights
    p1 <- ggplot(model_df, aes(x = Episode, y = ActorWeights)) +
      geom_line(color = "blue") +
      theme_minimal() +
      ggtitle("Actor Weights Progress over Episodes") +
      ylab("Actor Weights Value") +
      xlab("Episode")

    # Plot Actor Biases
    p2 <- ggplot(model_df, aes(x = Episode, y = ActorBiases)) +
      geom_line(color = "green") +
      theme_minimal() +
      ggtitle("Actor Biases Progress over Episodes") +
      ylab("Actor Biases Value") +
      xlab("Episode")

    # Plot Critic Weights
    p3 <- ggplot(model_df, aes(x = Episode, y = CriticWeights)) +
      geom_line(color = "red") +
      theme_minimal() +
      ggtitle("Critic Weights Progress over Episodes") +
      ylab("Critic Weights Value") +
      xlab("Episode")

    # Plot Critic Biases
    p4 <- ggplot(model_df, aes(x = Episode, y = CriticBiases)) +
      geom_line(color = "purple") +
      theme_minimal() +
      ggtitle("Critic Biases Progress over Episodes") +
      ylab("Critic Biases Value") +
      xlab("Episode")

    # Combine all plots into one grid
    grid.arrange(p1, p2, p3, p4, ncol = 2)
  } else {
    print("Warning: Length mismatch between weights and biases")
  }
}

# Example usage
print_model_summary("ppo_model_episode.json") # Print model summary
plot_learning_progress("ppo_model_episode.json") # Plot learning progress
