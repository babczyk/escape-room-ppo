# Install and load necessary libraries
if (!require(ggplot2)) install.packages("ggplot2")
if (!require(gridExtra)) install.packages("gridExtra")
if (!require(jsonlite)) install.packages("jsonlite")
if (!require(reshape2)) install.packages("reshape2")

library(ggplot2)
library(gridExtra)
library(jsonlite)
library(reshape2)

# Load JSON data from PPO model and progress
prog <- fromJSON("C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_prog.json") # nolint
save(prog, file = "C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_prog.RData") # nolint
load("C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_prog.RData") # nolint

model <- fromJSON("C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_model.json") # nolint
save(model, file = "C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_model.RData") # nolint
load("C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_model.RData") # nolint

# Helper function to create heatmaps
plot_heatmap <- function(weight_matrix, title) {
  weight_df <- melt(as.matrix(weight_matrix))
  ggplot(weight_df, aes(Var1, Var2, fill = value)) + # nolint
    geom_tile() +
    scale_fill_gradient(low = "blue", high = "red") +
    theme_minimal() +
    labs(title = title, x = "Input Nodes", y = "Output Nodes")
}

# 🔥 Heatmaps for policy and output layers
policy1_plot <- plot_heatmap(prog$POLICY1, "Policy Layer 1 Weights")
policy2_plot <- plot_heatmap(prog$POLICY2, "Policy Layer 2 Weights")
policy3_plot <- plot_heatmap(prog$POLICY3, "Policy Layer 3 Weights")
policyOutput_plot <- plot_heatmap(
  prog$POLICY_WEIGHTS_OUTPUT,
  "Policy Layer output Weights"
)


# 🔥 Plot Training Progress
reward_plot <- ggplot(
  data.frame(
    Episode = seq_along(model$RecentRewards),
    Reward = model$RecentRewards
  ),
  aes(x = Episode, y = Reward)
) +
  geom_line(color = "blue", size = 1) +
  theme_minimal() +
  labs(title = "Training Progress", x = "Episode", y = "Total Reward")

# 📊 Plot Bias Values Over Time
bias_df <- data.frame(
  Episode = seq_along(prog$POLICY_BIAS1),
  Bias1 = prog$POLICY_BIAS1,
  Bias2 = prog$POLICY_BIAS2,
  Bias3 = prog$POLICY_BIAS3
)

bias_plot <- ggplot(
  melt(bias_df, id = "Episode"),
  aes(x = Episode, y = value, color = variable)
) +
  geom_line(size = 1) +
  theme_minimal() +
  labs(title = "Bias Evolution Across Layers", x = "Episode", y = "Bias Value")


# Output bias
Output_bias_df <- data.frame(
  Episode = seq_along(prog$POLICY_OUTPUT_BIAS),
  OutputBias = prog$POLICY_OUTPUT_BIAS
)

Output_bias_plot <- ggplot(
  Output_bias_df,
  aes(x = Episode, y = OutputBias)
) +
  geom_line(color = "blue", size = 1) +
  theme_minimal() +
  labs(title = "Output Bias Evolution", x = "Episode", y = "Output Bias Value")

# 📌 Arrange all plots together
grid.arrange(policy1_plot,
  policy2_plot,
  policy3_plot,
  policyOutput_plot,
  reward_plot,
  bias_plot,
  Output_bias_plot,
  ncol = 3,
  nrow = 3
)
