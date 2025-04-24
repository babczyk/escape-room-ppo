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
}



# Example data: Let's assume your rewards are stored in a vector and times in another
times <- 1:100 # Example time points
rewards <- rnorm(100, mean = 50, sd = 10) # Example rewards, normally distributed

# Create a data frame
data <- data.frame(Time = times, Rewards = rewards)

# Plotting the rewards over time
ggplot(data, aes(x = Time, y = Rewards)) +
  geom_line(color = "blue") +
  geom_point(color = "red") +
  labs(title = "Rewards Over Time", x = "Time", y = "Rewards") +
  theme_minimal()
# Example usage
print_model_summary("ppo_model_episode.json") # Print model summary
plot_learning_progress("ppo_model_episode.json") # Plot learning progress
