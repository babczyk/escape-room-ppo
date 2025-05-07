# Load necessary libraries
library(jsonlite)
library(ggplot2)
library(dplyr)

# Load the JSON file
ppo_data <- fromJSON("PG_MAIN\\AI-project-escapeRoom\\statistics_R_files\\ppo_model_episode.json")

# Create data frame with episode numbers and rewards
ppo_df <- data.frame(
  episode = 1:ppo_data$IN_EPISODE,
  total_reward = ppo_data$TOTAL_Reward
)

# Plot total reward over episodes
reward_plot <- ggplot(ppo_df, aes(x = episode, y = total_reward)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_smooth(method = "loess", color = "red", se = FALSE, linetype = "dashed") +
  labs(
    title = "PPO Learning Progress: Total Reward Over Episodes",
    x = "Episode",
    y = "Total Reward"
  ) +
  theme_minimal(base_size = 14)

print(reward_plot)
