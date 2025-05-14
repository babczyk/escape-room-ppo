# Load libraries
library(jsonlite)
library(ggplot2)
library(reshape2)
library(gridExtra)

# Load JSON data
ppo_data <- fromJSON("PG_MAIN\\AI-project-escapeRoom\\statistics_R_files\\ppo_model_episode.json")

# --- Plot 1: Total reward over episodes ---
ppo_df <- data.frame(
  episode = 1:ppo_data$IN_EPISODE,
  total_reward = ppo_data$TOTAL_Reward
)

reward_plot <- ggplot(ppo_df, aes(x = episode, y = total_reward)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_smooth(method = "loess", color = "red", se = FALSE, linetype = "dashed") +
  labs(
    title = "PPO Learning Progress: Total Reward Over Episodes",
    x = "Episode",
    y = "Total Reward"
  ) +
  theme_minimal(base_size = 14)

# --- Helper: Plot a heatmap for a matrix ---
plot_heatmap <- function(matrix_data, title) {
  melted <- melt(matrix_data)
  ggplot(melted, aes(x = Var2, y = Var1, fill = value)) +
    geom_tile() +
    scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    labs(title = title, x = "Neuron/Unit", y = "Layer/Input") +
    theme_minimal(base_size = 10)
}

# --- Plot 2: Actor weights (first layer) ---
actor_w1 <- ppo_data$IN_ACTOR_WEIGHTS[[1]]
actor_w1_plot <- plot_heatmap(actor_w1, "Actor Weights Layer 1")

# --- Plot 3: Actor biases (first layer) ---
actor_b1 <- matrix(ppo_data$IN_ACTOR_BIASES[[1]], nrow = 1)
actor_b1_plot <- plot_heatmap(actor_b1, "Actor Biases Layer 1")

# --- Plot 4: Critic weights (first layer) ---
critic_w1 <- ppo_data$IN_CRITIC_WEIGHTS[[1]]
critic_w1_plot <- plot_heatmap(critic_w1, "Critic Weights Layer 1")

# --- Plot 5: Critic biases (first layer) ---
critic_b1 <- matrix(ppo_data$IN_CRITIC_BIASES[[1]], nrow = 1)
critic_b1_plot <- plot_heatmap(critic_b1, "Critic Biases Layer 1")

# --- Display all plots ---
# You may need to adjust ncol/nrow depending on your screen
grid.arrange(reward_plot, actor_w1_plot, actor_b1_plot, critic_w1_plot, critic_b1_plot, ncol = 2)
