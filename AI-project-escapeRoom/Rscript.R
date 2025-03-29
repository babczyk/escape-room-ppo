install.packages("jsonlite")
library(jsonlite)

prog <- fromJSON("C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_prog.json") # nolint
save(prog, file = "C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_prog.RData") # nolint
load("C:\\Users\\ronin\\OneDrive\\Desktop\\stuff\\escapeRoomAi\\PG_MAIN\\AI-project-escapeRoom\\bin\\Debug\\net8.0\\ppo_prog.RData") # nolint

model <- fromJSON("C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_model.json") # nolint
save(model, file = "C://Users/ronin/OneDrive/Desktop/stuff/escapeRoomAi/PG_MAIN/AI-project-escapeRoom/bin/Debug/net8.0/ppo_model.RData") # nolint
load("C:\\Users\\ronin\\OneDrive\\Desktop\\stuff\\escapeRoomAi\\PG_MAIN\\AI-project-escapeRoom\\bin\\Debug\\net8.0\\ppo_model.RData") # nolint

library(ggplot2)
library(reshape2)
library(gridExtra)

# Convert JSON weights to a matrix
policy_weights <- as.matrix(prog$POLICY_WEIGHTS_OUTPUT)

# Convert matrix to a dataframe for plotting
policy_df <- melt(policy_weights)

# 🔥 Plot 1: Heatmap
heatmap_plot <- ggplot(policy_df, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "blue", high = "red") +
  theme_minimal() +
  labs(
    title = "Heatmap of Policy Network Weights",
    x = "Input Nodes", y = "Output Nodes"
  )

# 🔥 Plot 2: Training Progress
reward_plot <- ggplot(
  data.frame(
    Episode = 1:seq_along(model$RecentRewards),
    Reward = model$RecentRewards
  ),
  aes(x = Episode, y = Reward)
) +
  geom_line(color = "blue", size = 1) +
  theme_minimal() +
  labs(title = "Training Progress", x = "Episode", y = "Total Reward")

# 📌 Arrange both plots together
grid.arrange(heatmap_plot, reward_plot, ncol = 2)
