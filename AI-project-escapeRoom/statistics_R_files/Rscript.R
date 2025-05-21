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
    subtitle = "PPO Total Reward Over Episodes",
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

# --- Plot 6: Actor weights (second layer) ---
actor_w2 <- ppo_data$IN_ACTOR_WEIGHTS[[2]]
actor_w2_plot <- plot_heatmap(actor_w2, "Actor Weights Layer 2")

# --- Plot 7: Actor biases (second layer) ---
actor_b2 <- matrix(ppo_data$IN_ACTOR_BIASES[[2]], nrow = 1)
actor_b2_plot <- plot_heatmap(actor_b2, "Actor Biases Layer 2")

# --- Plot 8: Critic weights (second layer) ---
critic_w2 <- ppo_data$IN_CRITIC_WEIGHTS[[2]]
critic_w2_plot <- plot_heatmap(critic_w2, "Critic Weights Layer 2")

# --- Plot 9: Critic biases (second layer) ---
critic_b2 <- matrix(ppo_data$IN_CRITIC_BIASES[[2]], nrow = 1)
critic_b2_plot <- plot_heatmap(critic_b2, "Critic Biases Layer 2")

# --- Plot 10: Actor weights (third layer) ---
actor_w3 <- ppo_data$IN_ACTOR_WEIGHTS[[3]]
actor_w3_plot <- plot_heatmap(actor_w3, "Actor Weights Layer 3")

# --- Plot 11: Actor biases (third layer) ---
actor_b3 <- matrix(ppo_data$IN_ACTOR_BIASES[[3]], nrow = 1)
actor_b3_plot <- plot_heatmap(actor_b3, "Actor Biases Layer 3")

# --- Display all plots ---
# You may need to adjust ncol/nrow depending on your screen
grid.arrange(
  reward_plot, actor_w1_plot, actor_b1_plot, critic_w1_plot, critic_b1_plot,
  actor_w2_plot, actor_b2_plot, critic_w2_plot, critic_b2_plot,
  actor_w3_plot, actor_b3_plot,
  ncol = 3
)
