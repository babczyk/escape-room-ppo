using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Microsoft.Xna.Framework;

class PPO
{
    // Neural network architecture
    private const int HIDDEN_LAYER_1_SIZE = 128;
    private const int HIDDEN_LAYER_2_SIZE = 64;
    private const int HIDDEN_LAYER_3_SIZE = 32;

    // Network weights
    private double[,] policyWeights1;
    private double[,] policyWeights2;
    private double[,] policyWeights3;
    private double[] policyOutputWeights;

    private double[,] valueWeights1;
    private double[,] valueWeights2;
    private double[,] valueWeights3;
    private double[] valueOutputWeights;

    // Hyperparameters
    private const double GAMMA = 0.99f;
    private const double CLIP_EPSILON = 0.2f;
    private const double LEARNING_RATE = 0.0003f;
    private const int EPOCHS = 4;
    private const double ENTROPY_COEF = 0.01f;
    private const double VALUE_COEF = 0.5f;
    private const int BATCH_SIZE = 64;

    // Training metrics
    private List<double> episodeRewards;
    private List<double> policyLosses;
    private List<double> valueLosses;
    private List<double> entropyValues;

    private Random random;
    private int stateSize;
    private int actionSize;


    /// <summary>
    /// Initializes a new instance of the PPO class.
    /// Sets up the neural networks for both policy and value functions.
    /// </summary>
    public PPO(int stateSize = 6, int actionSize = 5)
    {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        random = new Random();

        // Initialize policy network
        policyWeights1 = InitializeWeights(stateSize, HIDDEN_LAYER_1_SIZE);
        policyWeights2 = InitializeWeights(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
        policyWeights3 = InitializeWeights(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE);
        policyOutputWeights = InitializeWeights(HIDDEN_LAYER_3_SIZE, actionSize).Cast<double>().ToArray();

        // Initialize value network
        valueWeights1 = InitializeWeights(stateSize, HIDDEN_LAYER_1_SIZE);
        valueWeights2 = InitializeWeights(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
        valueWeights3 = InitializeWeights(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE);
        valueOutputWeights = InitializeWeights(HIDDEN_LAYER_3_SIZE, 1).Cast<double>().ToArray();

        // Initialize metrics
        episodeRewards = new List<double>();
        policyLosses = new List<double>();
        valueLosses = new List<double>();
        entropyValues = new List<double>();
    }

    /// <summary>
    /// Initializes a weight matrix with scaled random values.
    /// </summary>
    /// <param name="inputSize">Size of the input layer</param>
    /// <param name="outputSize">Size of the output layer</param>
    /// <returns>A 2D array of initialized weights scaled by sqrt(2/inputSize)</returns>
    private double[,] InitializeWeights(int inputSize, int outputSize)
    {
        double[,] weights = new double[inputSize, outputSize];
        double scale = Math.Sqrt(2.0 / inputSize);

        for (int i = 0; i < inputSize; i++)
            for (int j = 0; j < outputSize; j++)
                weights[i, j] = (random.NextDouble() * 2 - 1) * scale;

        return weights;
    }

    /// <summary>
    /// Performs a linear transformation (matrix multiplication) of the input.
    /// </summary>
    /// <param name="input">Input vector</param>
    /// <param name="weights">Weight matrix</param>
    /// <param name="biasWeights">Optional bias weights</param>
    /// <returns>Output vector after linear transformation</returns>
    private double[] LinearLayer(double[] input, double[,] weights, double[]? biasWeights = null)
    {
        int outputSize = weights.GetLength(1);
        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < input.Length; j++)
                output[i] += input[j] * weights[j, i];

            if (biasWeights != null)
                output[i] += biasWeights[i];
        }

        return output;
    }

    /// <summary>
    /// Applies the ReLU activation function element-wise to the input vector.
    /// ReLU(x) = max(0, x)
    /// </summary>
    /// <param name="x">Input vector</param>
    /// <returns>Vector with ReLU activation applied</returns>
    private double[] ReLU(double[] x)
    {
        return x.Select(v => Math.Max(0, v)).ToArray();
    }

    /// <summary>
    /// Applies the softmax function to convert logits to probabilities.
    /// </summary>
    /// <param name="x">Input logits</param>
    /// <returns>Probability distribution that sums to 1</returns>
    private double[] Softmax(double[] x)
    {
        double max = x.Max();
        double[] exp = x.Select(v => Math.Exp(v - max)).ToArray();
        double sum = exp.Sum();
        return exp.Select(v => v / sum).ToArray();
    }

    /// <summary>
    /// Computes advantages using Generalized Advantage Estimation (GAE).
    /// </summary>
    /// <param name="rewards">List of rewards from the episode</param>
    /// <param name="values">List of estimated state values</param>
    /// <returns>List of normalized advantages</returns>
    private List<double> ComputeAdvantages(List<Double> rewards, List<double> values)
    {
        List<double> advantages = new List<double>();
        double nextValue = 0;
        double advantage = 0;

        for (int t = rewards.Count - 1; t >= 0; t--)
        {
            double delta = rewards[t] + GAMMA * nextValue - values[t];
            advantage = delta + GAMMA * 0.95f * advantage;
            advantages.Insert(0, advantage);
            nextValue = values[t];
        }

        // Normalize advantages
        if (advantages.Count > 0)
        {
            double mean = advantages.Average();
            double std = Math.Sqrt(advantages.Select(x => Math.Pow(x - mean, 2)).Average() + 1e-8);
            return advantages.Select(a => (a - mean) / std).ToList();
        }
        return advantages;
    }

    private double[] PolicyForward(double[] input)
    {
        var layer1 = ReLU(LinearLayer(input, policyWeights1));
        var layer2 = ReLU(LinearLayer(layer1, policyWeights2));
        var layer3 = ReLU(LinearLayer(layer2, policyWeights3));
        var output = LinearLayer(layer3, new double[HIDDEN_LAYER_3_SIZE, actionSize], policyOutputWeights);
        return Softmax(output);
    }

    private double ValueForward(double[] input)
    {
        var layer1 = ReLU(LinearLayer(input, valueWeights1));
        var layer2 = ReLU(LinearLayer(layer1, valueWeights2));
        var layer3 = ReLU(LinearLayer(layer2, valueWeights3));
        var output = LinearLayer(layer3, new double[HIDDEN_LAYER_3_SIZE, 1], valueOutputWeights);
        return output[0];
    }

    private (TrajectoryData trajectory, double totalReward) CollectTrajectory(GameEnvironment env)
    {
        var trajectory = new TrajectoryData();
        double totalReward = 0;
        double[] state = env.GetState();
        bool done = false;

        while (!done)
        {
            double[] stateVector = state.Select(s => (double)s).ToArray();
            double[] actionProbs = PolicyForward(stateVector);
            int action = SampleAction(actionProbs);
            double actionProb = actionProbs[action];
            double valueEstimate = ValueForward(stateVector);
            trajectory.AddStep(stateVector, action, actionProb, valueEstimate);

            var (nextState, reward, isDone) = env.Step(action);
            totalReward += reward;
            trajectory.rewards.Add(reward);

            if (isDone)
            {
                done = true;
                trajectory.advantages = ComputeAdvantages(trajectory.rewards, trajectory.values);
            }

            state = nextState;
            Thread.Sleep(10); // Visualization delay
        }

        return (trajectory, totalReward);
    }

    public void Train(GameEnvironment env, int episodes)
    {
        double bestReward = double.MinValue;
        double averageReward = 0;
        int episodesSinceImprovement = 0;

        for (int episode = 0; episode < episodes; episode++)
        {
            var (trajectory, totalReward) = CollectTrajectory(env);

            // Update networks multiple times with the collected data
            for (int epoch = 0; epoch < EPOCHS; epoch++)
            {
                var batchIndices = Enumerable.Range(0, trajectory.states.Count)
                    .OrderBy(x => random.Next())
                    .ToList();

                for (int i = 0; i < batchIndices.Count; i += BATCH_SIZE)
                {
                    var batchSlice = batchIndices.Skip(i).Take(BATCH_SIZE);
                    UpdateNetworksBatch(trajectory, batchSlice.ToList());
                }
            }

            // Track metrics
            episodeRewards.Add(totalReward);
            averageReward = episodeRewards.TakeLast(100).Average();

            // Log progress
            if (episode % 10 == 0)
            {
                Console.WriteLine($"Episode {episode}:");
                Console.WriteLine($"Total Reward: {totalReward:F2}");
                Console.WriteLine($"Average Reward (last 100): {averageReward:F2}");
                Console.WriteLine($"Policy Loss: {policyLosses.LastOrDefault():F4}");
                Console.WriteLine($"Value Loss: {valueLosses.LastOrDefault():F4}");
                Console.WriteLine($"Entropy: {entropyValues.LastOrDefault():F4}");
                Console.WriteLine("--------------------");
            }

            // Save best model
            if (averageReward > bestReward)
            {
                bestReward = averageReward;
                SaveModel("best_model.json");
                episodesSinceImprovement = 0;
            }
            else
            {
                episodesSinceImprovement++;
            }

            // Early stopping
            if (episodesSinceImprovement > 900)
            {
                Console.WriteLine("Early stopping triggered - No improvement for 200 episodes");
                break;
            }
        }
    }
    /// <summary>
    /// Loads a model from a JSON file.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="filePath"></param>
    /// <returns></returns>
    private void LoadModel(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException("Model file not found.");

        string json = File.ReadAllText(filePath);
        var model = JsonSerializer.Deserialize<dynamic>(json);


    }

    /// <summary>
    /// Loads the policy and value networks from a JSON file.
    /// </summary>
    /// <param name="filePath"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public (int episode, double bestReward, List<int> recentRewards) LoadProgress(string filePath)
    {
        string json = File.ReadAllText(filePath);
        var progress = JsonSerializer.Deserialize<JsonElement>(json);
        Console.WriteLine(progress);

        // Accessing specific properties:
        if (progress.TryGetProperty("Episode", out var episodeProperty))
        {
            int episode = episodeProperty.GetInt32();
            Console.WriteLine($"Episode: {episode}");
        }
        else
        {
            throw new InvalidOperationException("Failed to find 'Episode' in the deserialized JSON.");
        }


        var episodeA = progress.GetProperty("Episode").GetInt32();
        var bestReward = progress.GetProperty("BestReward").GetDouble();
        var recentRewards = progress.GetProperty("RecentRewards")
                                    .EnumerateArray()
                                    .Select(e => e.GetInt32())
                                    .ToList();

        return (episodeA, bestReward, recentRewards);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="filePath"></param>
    /// <param name="model"></param>
    private void SaveModel(string filePath)
    {
        var model = new
        {

        };

        string json = JsonSerializer.Serialize(model, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Saves the current training progress to a JSON file.
    /// </summary>
    /// <param name="filePath"></param>
    /// <param name="episode"></param>
    /// <param name="bestReward"></param>
    /// <param name="recentRewards"></param>
    public void SaveProgress(string filePath, int episode, double bestReward, List<int> recentRewards)
    {
        var progress = new
        {
            Episode = episode,
            BestReward = bestReward,
            RecentRewards = recentRewards
        };
        var json = JsonSerializer.Serialize(progress);
        File.WriteAllText(filePath, json);
    }

    /// <summary>
    /// Samples an action from a probability distribution over actions.
    /// </summary>
    /// <param name="actionProbs">Probability distribution over actions</param>
    /// <returns>Chosen action index</returns>
    private int SampleAction(double[] actionProbs)
    {
        double sample = random.NextDouble();
        double sum = 0;

        for (int i = 0; i < actionProbs.Length; i++)
        {
            sum += actionProbs[i];
            if (sample <= sum)
                return i;
        }

        return actionProbs.Length - 1;
    }


    private void UpdateNetworkWeights(double totalLoss)
    {
        // Calculate gradients and update policy network weights
        UpdatePolicyNetworkWeights(totalLoss);

        // Calculate gradients and update value network weights
        UpdateValueNetworkWeights(totalLoss);
    }

    private void UpdatePolicyNetworkWeights(double loss)
    {
        // Learning rate with decay
        double currentLearningRate = LEARNING_RATE * (1.0 / (1.0 + 0.0001 * episodeRewards.Count));

        // Update policy network layer 1
        for (int i = 0; i < policyWeights1.GetLength(0); i++)
        {
            for (int j = 0; j < policyWeights1.GetLength(1); j++)
            {
                // Gradient descent update with momentum
                double gradient = loss * CalculateLayerGradient(policyWeights1[i, j]);
                double momentum = 0.9 * policyWeights1[i, j];
                double update = currentLearningRate * (gradient + momentum);

                // Update weight with gradient clipping
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeights1[i, j] -= update;

                // Store gradient for momentum
                policyWeights1[i, j] = gradient;
            }
        }

        // Update policy network layer 2
        for (int i = 0; i < policyWeights2.GetLength(0); i++)
        {
            for (int j = 0; j < policyWeights2.GetLength(1); j++)
            {
                double gradient = loss * CalculateLayerGradient(policyWeights2[i, j]);
                double momentum = 0.9 * policyWeights2[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeights2[i, j] -= update;
                policyWeights2[i, j] = gradient;
            }
        }

        // Update policy network layer 3
        for (int i = 0; i < policyWeights3.GetLength(0); i++)
        {
            for (int j = 0; j < policyWeights3.GetLength(1); j++)
            {
                double gradient = loss * CalculateLayerGradient(policyWeights3[i, j]);
                double momentum = 0.9 * policyWeights3[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeights3[i, j] -= update;
                policyWeights3[i, j] = gradient;
            }
        }

        // Update policy output weights
        for (int i = 0; i < policyOutputWeights.Length; i++)
        {
            double gradient = loss * CalculateOutputGradient(policyOutputWeights[i]);
            double momentum = 0.9 * policyOutputWeights[i];
            double update = currentLearningRate * (gradient + momentum);
            update = Math.Clamp(update, -1.0, 1.0);
            policyOutputWeights[i] -= update;
            policyOutputWeights[i] = gradient;
        }
    }

    private void UpdateValueNetworkWeights(double loss)
    {
        // Similar structure to policy network updates but for value network
        double currentLearningRate = LEARNING_RATE * (1.0 / (1.0 + 0.0001 * episodeRewards.Count));

        // Update value network weights with similar pattern...
        // (Implementation similar to policy network updates)

        // Example for first layer:
        for (int i = 0; i < valueWeights1.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeights1.GetLength(1); j++)
            {
                double gradient = loss * CalculateLayerGradient(valueWeights1[i, j]);
                double momentum = 0.9 * valueWeights1[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                valueWeights1[i, j] -= update;
                valueWeights1[i, j] = gradient;
            }
        }

        // Continue with other layers...
    }

    private double CalculateLayerGradient(double weight)
    {
        // Add small noise for exploration
        double noise = random.NextDouble() * 0.01;

        // Calculate gradient with weight decay
        double weightDecay = 0.0001 * weight;

        return weight + noise + weightDecay;
    }

    private double CalculateOutputGradient(double weight)
    {
        // Similar to layer gradient but with different scaling
        double noise = random.NextDouble() * 0.005;
        double weightDecay = 0.0001 * weight;

        return weight + noise + weightDecay;
    }
    /// <summary>
    /// Updates both policy and value networks using collected trajectory data.
    /// </summary>
    /// <param name="states">List of state vectors</param>
    /// <param name="actions">List of actions taken</param>
    /// <param name="advantages">List of computed advantages</param>
    /// <param name="oldActionProbs">List of action probabilities from old policy</param>
    /// <param name="values">List of estimated state values</param>
    /// <param name="rewards">List of rewards received</param>
    private void UpdateNetworksBatch(TrajectoryData trajectory, List<int> batchIndices)
    {
        double policyLoss = 0;
        double valueLoss = 0;
        double entropy = 0;

        foreach (int idx in batchIndices)
        {
            var currentProbs = PolicyForward(trajectory.states[idx]);
            double ratio = currentProbs[trajectory.actions[idx]] / trajectory.oldActionProbs[idx];

            // Policy loss with clipping
            double advantage = trajectory.advantages[idx];
            double unclippedLoss = ratio * advantage;
            double clippedLoss = Math.Clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantage;
            policyLoss += -Math.Min(unclippedLoss, clippedLoss);

            // Value loss
            double returns = advantage + trajectory.values[idx];
            double valueEstimate = ValueForward(trajectory.states[idx]);
            valueLoss += Math.Pow(valueEstimate - returns, 2);

            // Entropy for exploration
            entropy += -currentProbs.Sum(p => p * Math.Log(Math.Max(p, 1e-10)));
        }

        // Average losses
        int batchSize = batchIndices.Count;
        policyLoss /= batchSize;
        valueLoss /= batchSize;
        entropy /= batchSize;

        // Track metrics
        System.Console.WriteLine(policyLoss);
        System.Console.WriteLine(valueLoss);
        policyLosses.Add(policyLoss);
        valueLosses.Add(valueLoss);
        entropyValues.Add(entropy);

        // Apply updates
        double totalLoss = policyLoss + VALUE_COEF * valueLoss - ENTROPY_COEF * entropy;
        UpdateNetworkWeights(totalLoss);
    }

    /// <summary>
    /// Updates the policy network weights using gradient descent.
    /// </summary>
    /// <param name="state">Current state vector</param>
    /// <param name="action">Taken action</param>
    /// <param name="loss">Computed policy loss</param>
    private void UpdatePolicyWeights(double[] state, int action, double loss)
    {
        // Simple gradient descent update
        for (int i = 0; i < policyWeights1.GetLength(0); i++)
            for (int j = 0; j < policyWeights1.GetLength(1); j++)
                policyWeights1[i, j] -= LEARNING_RATE * loss * state[i];

        for (int i = 0; i < policyWeights2.Length; i++)
            policyOutputWeights[i] -= LEARNING_RATE * loss;
    }

    /// <summary>
    /// Updates the value network weights using gradient descent.
    /// </summary>
    /// <param name="state">Current state vector</param>
    /// <param name="loss">Computed value loss</param>
    private void UpdateValueWeights(double[] state, double loss)
    {
        for (int i = 0; i < valueWeights1.GetLength(0); i++)
            for (int j = 0; j < valueWeights1.GetLength(1); j++)
                valueWeights1[i, j] -= LEARNING_RATE * loss * state[i];

        for (int i = 0; i < valueWeights2.Length; i++)
            valueOutputWeights[i] -= LEARNING_RATE * loss;
    }

}

class TrajectoryData
{
    public List<double[]> states = new List<double[]>();
    public List<int> actions = new List<int>();
    public List<double> oldActionProbs = new List<double>();
    public List<double> values = new List<double>();
    public List<double> rewards = new List<double>();
    public List<double> advantages = new List<double>();

    public void AddStep(double[] state, int action, double actionProb, double value)
    {
        states.Add(state);
        actions.Add(action);
        oldActionProbs.Add(actionProb);
        values.Add(value);
    }
}