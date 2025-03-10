#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Microsoft.Xna.Framework;

class PPO
{
    private static Random rng = new Random();

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

    // Weight gradients and momentum for proper updates
    private double[,] policyGradients1;
    private double[,] policyGradients2;
    private double[,] policyGradients3;
    private double[] policyOutputGradients;

    private double[,] valueGradients1;
    private double[,] valueGradients2;
    private double[,] valueGradients3;
    private double[] valueOutputGradients;

    // Hyperparameters
    private const double GAMMA = 0.99f;
    private const double CLIP_EPSILON = 0.2f;
    private const double LEARNING_RATE = 0.0001f;
    private const int EPOCHS = 4;
    private double ENTROPY_COEF = 0.01f; // Made non-constant to allow decay
    private const double VALUE_COEF = 0.5f;
    private const int BATCH_SIZE = 64;

    // Training metrics
    private List<double> episodeRewards;
    private List<double> policyLosses;
    private List<double> valueLosses;
    private List<double> entropyValues;

    // BatchNorm running statistics
    private double[] runningMean;
    private double[] runningVar;
    private const double MOMENTUM = 0.99;

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

        // Initialize gradients and momentum
        policyGradients1 = new double[stateSize, HIDDEN_LAYER_1_SIZE];
        policyGradients2 = new double[HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE];
        policyGradients3 = new double[HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE];
        policyOutputGradients = new double[HIDDEN_LAYER_3_SIZE];

        valueGradients1 = new double[stateSize, HIDDEN_LAYER_1_SIZE];
        valueGradients2 = new double[HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE];
        valueGradients3 = new double[HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE];
        valueOutputGradients = new double[HIDDEN_LAYER_3_SIZE];

        // Initialize BatchNorm statistics
        runningMean = new double[stateSize];
        runningVar = new double[stateSize];
        for (int i = 0; i < stateSize; i++)
        {
            runningVar[i] = 1.0;
        }

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
    private double[] ReLU(double[] x, double alpha = 0.01)
    {
        return x.Select(v => v >= 0 ? v : alpha * v).ToArray();
    }

    /// <summary>
    /// Applies the softmax function to convert logits to probabilities.
    /// </summary>
    /// <param name="x">Input logits</param>
    /// <returns>Probability distribution that sums to 1</returns>
    private double[] Softmax(double[] logits)
    {
        // Normalize logits to avoid numerical instability
        double mean = logits.Average();
        double std = Math.Sqrt(logits.Select(l => Math.Pow(l - mean, 2)).Average());

        // Normalize to have zero mean and unit variance (if std > 0)
        double[] normalizedLogits = std > 0 ? logits.Select(l => (l - mean) / std).ToArray() : logits;

        double maxLogit = normalizedLogits.Max();
        double[] expValues = normalizedLogits.Select(l => Math.Exp(l - maxLogit)).ToArray();
        double sumExp = expValues.Sum();
        return expValues.Select(e => e / sumExp).ToArray();
    }

    private double[] PolicyForward(double[] input)
    {
        var layer1 = ReLU(LinearLayer(input, policyWeights1));
        var layer2 = ReLU(LinearLayer(layer1, policyWeights2));

        // BatchNorm to stabilize training
        layer2 = BatchNorm(layer2);

        var layer3 = ReLU(LinearLayer(layer2, policyWeights3));
        var output = LinearLayer(layer3, new double[HIDDEN_LAYER_3_SIZE, actionSize], policyOutputWeights);
        //Console.WriteLine($"Logits: {string.Join(", ", output)}");

        // Scale logits before Softmax to avoid uniform distributions
        for (int i = 0; i < output.Length; i++)
        {
            output[i] *= RandomGaussian(1, 0.1); ;   // Adjust scaling factor if needed
        }
        var outputReturn = Softmax(output);
        if (outputReturn.Contains(Double.NaN) || outputReturn.Contains(Double.PositiveInfinity))
        {
            Console.WriteLine("Logits or Softmax output contains NaN or Inf");
        }
        return Softmax(outputReturn);
    }

    public static double RandomGaussian(double mean = 0.0, double stdDev = 1.0)
    {
        double u1 = 1.0 - rng.NextDouble(); // Uniform(0,1] random value
        double u2 = 1.0 - rng.NextDouble();

        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal; // Scale to desired mean/stdDev
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

    private double[] BatchNorm(double[] input, double epsilon = 1e-5)
    {
        double mean = input.Average();
        double variance = input.Select(x => Math.Pow(x - mean, 2)).Average();
        return input.Select(x => (x - mean) / Math.Sqrt(variance + epsilon)).ToArray();
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
        }
        return (trajectory, totalReward);
    }

    public void Train(GameEnvironment env, int episodes, string modelPath, string progressPath)
    {
        double bestReward = double.MinValue;
        double averageReward = 0;
        int episodesSinceImprovement = 0;
        int episode = 0;
        if (File.Exists(progressPath) && File.Exists(modelPath))
        {
            var progress = LoadProgress(progressPath);
            episode = progress.episode;
            bestReward = progress.bestReward;
            LoadModel(modelPath);
        }
        for (; episode < episodes; episode++)
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
            averageReward = episodeRewards.TakeLast(5).Average(); //take an avrege of 50 episodes (50 generations)
            Console.WriteLine($"avrege reward {averageReward}:");
            // Log progress
            if (episode % 5 == 0)
            {
                Console.WriteLine($"Episode {episode}:");
                Console.WriteLine($"Total Reward: {totalReward:F2}");
                Console.WriteLine($"Average Reward: {averageReward:F2}");
                Console.WriteLine($"Policy Loss: {policyLosses.LastOrDefault():F4}");
                Console.WriteLine($"Value Loss: {valueLosses.LastOrDefault():F4}");
                Console.WriteLine($"Entropy: {entropyValues.LastOrDefault():F4}");
                Console.WriteLine("--------------------");
                SaveProgress(progressPath, episode, bestReward, episodeRewards);
            }

            // Save best model
            if (averageReward > bestReward)
            {
                bestReward = averageReward;
                Console.WriteLine("Saving best model...");
                SaveModel(modelPath, episode);
                LoadModel(modelPath);
                episodesSinceImprovement = 0;
            }
            else
            {
                episodesSinceImprovement++;
            }

            // Early stopping
            if (episodesSinceImprovement > 200)
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
            throw new FileNotFoundException($"Model file not found at {filePath}");

        try
        {
            string json = File.ReadAllText(filePath);

            // Use JsonDocument for more flexible parsing
            using (JsonDocument document = JsonDocument.Parse(json))
            {
                var root = document.RootElement;

                // Get policy weights (use case-insensitive property matching)
                var policy1Property = GetProperty(root, "POLICY1");
                var policy2Property = GetProperty(root, "POLICY2");
                var policy3Property = GetProperty(root, "POLICY3");
                var policyOutputProperty = GetProperty(root, "POLICY_OUTPUT");

                // Get value weights
                var value1Property = GetProperty(root, "VALUE1");
                var value2Property = GetProperty(root, "VALUE2");
                var value3Property = GetProperty(root, "VALUE3");
                var valueOutputProperty = GetProperty(root, "VALUE_OUTPUT");

                // Deserialize with appropriate types
                policyWeights1 = ConvertTo2DArray(JsonSerializer.Deserialize<double[][]>(policy1Property.GetRawText()));
                policyWeights2 = ConvertTo2DArray(JsonSerializer.Deserialize<double[][]>(policy2Property.GetRawText()));
                policyWeights3 = ConvertTo2DArray(JsonSerializer.Deserialize<double[][]>(policy3Property.GetRawText()));
                policyOutputWeights = JsonSerializer.Deserialize<double[]>(policyOutputProperty.GetRawText());

                valueWeights1 = ConvertTo2DArray(JsonSerializer.Deserialize<double[][]>(value1Property.GetRawText()));
                valueWeights2 = ConvertTo2DArray(JsonSerializer.Deserialize<double[][]>(value2Property.GetRawText()));
                valueWeights3 = ConvertTo2DArray(JsonSerializer.Deserialize<double[][]>(value3Property.GetRawText()));
                valueOutputWeights = JsonSerializer.Deserialize<double[]>(valueOutputProperty.GetRawText());
            }

            Console.WriteLine($"Model successfully loaded from {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading model: {ex.Message}");
            Console.WriteLine(ex.StackTrace);
            throw; // Re-throw the exception to make sure it doesn't fail silently
        }
    }

    // Helper method to get property with case-insensitive matching
    private JsonElement GetProperty(JsonElement element, string propertyName)
    {
        foreach (var property in element.EnumerateObject())
        {
            if (string.Equals(property.Name, propertyName, StringComparison.OrdinalIgnoreCase))
            {
                return property.Value;
            }
        }

        throw new KeyNotFoundException($"Property '{propertyName}' not found in JSON");
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
        string absolutePath = Path.GetFullPath(filePath);

        Console.WriteLine($"Absolute Path: {absolutePath}");
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
    private void SaveModel(string filePath, int episode = 1)
    {
        var model = new
        {
            IN_EPESODE = episode,
            POLICY1 = ConvertToJaggedArray(policyWeights1),
            POLICY2 = ConvertToJaggedArray(policyWeights2),
            POLICY3 = ConvertToJaggedArray(policyWeights3),
            POLICY_OUTPUT = policyOutputWeights,

            VALUE1 = ConvertToJaggedArray(valueWeights1),
            VALUE2 = ConvertToJaggedArray(valueWeights2),
            VALUE3 = ConvertToJaggedArray(valueWeights3),
            VALUE_OUTPUT = valueOutputWeights


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
    public void SaveProgress(string filePath, int episode, double bestReward, List<double> recentRewards)
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

    private static double[][] ConvertToJaggedArray(double[,] array)
    {
        int rows = array.GetLength(0);
        int cols = array.GetLength(1);
        double[][] jaggedArray = new double[rows][];
        for (int i = 0; i < rows; i++)
        {
            jaggedArray[i] = new double[cols];
            for (int j = 0; j < cols; j++)
            {
                jaggedArray[i][j] = array[i, j];
            }
        }
        return jaggedArray;
    }

    private static double[,] ConvertTo2DArray(double[][] jaggedArray)
    {
        int rows = jaggedArray.Length;
        int cols = jaggedArray[0].Length;
        double[,] array = new double[rows, cols];
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                array[i, j] = jaggedArray[i][j];
            }
        }
        return array;
    }
    /// <summary>
    /// Samples an action from a probability distribution over actions.
    /// </summary>
    /// <param name="actionProbs">Probability distribution over actions</param>
    /// <returns>Chosen action index</returns>
    private int SampleAction(double[] actionProbs)
    {
        // Check if probabilities are valid
        double sum = actionProbs.Sum();
        if (Math.Abs(sum - 1.0) > 1e-3 || actionProbs.Any(p => p < 0 || p > 1))
        {
            Console.WriteLine($"Warning: Invalid probability distribution. Sum: {sum}");
            // Normalize probabilities if they don't sum to 1
            actionProbs = actionProbs.Select(p => Math.Max(0, p) / actionProbs.Sum(q => Math.Max(0, q))).ToArray();
        }

        double sample = random.NextDouble();
        double cumSum = 0;

        for (int i = 0; i < actionProbs.Length; i++)
        {
            cumSum += actionProbs[i];
            if (sample <= cumSum)
                return i;
        }

        // Fallback to highest probability action
        return Array.IndexOf(actionProbs, actionProbs.Max());
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

    /// <summary>
    /// Updates the weights of the value network using gradient descent with momentum.
    /// This function applies a learning rate decay and gradient clipping to stabilize training.
    /// </summary>
    /// <param name="loss">The loss value used to compute the gradient updates.</param>
    private void UpdateValueNetworkWeights(double loss)
    {
        // Compute the current learning rate with decay
        double currentLearningRate = LEARNING_RATE * (1.0 / (1.0 + 0.0001 * episodeRewards.Count));

        // Update value network layer 1
        for (int i = 0; i < valueWeights1.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeights1.GetLength(1); j++)
            {
                // Compute the gradient and apply momentum
                double gradient = loss * CalculateLayerGradient(valueWeights1[i, j]);
                double momentum = 0.9 * valueWeights1[i, j];
                double update = currentLearningRate * (gradient + momentum);

                // Apply gradient clipping
                update = Math.Clamp(update, -1.0, 1.0);

                // Update weights
                valueWeights1[i, j] -= update;

                // Store gradient for future momentum updates
                valueWeights1[i, j] = gradient;
            }
        }

        // Update value network layer 2
        for (int i = 0; i < valueWeights2.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeights2.GetLength(1); j++)
            {
                double gradient = loss * CalculateLayerGradient(valueWeights2[i, j]);
                double momentum = 0.9 * valueWeights2[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                valueWeights2[i, j] -= update;
                valueWeights2[i, j] = gradient;
            }
        }

        // Update value network layer 3
        for (int i = 0; i < valueWeights3.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeights3.GetLength(1); j++)
            {
                double gradient = loss * CalculateLayerGradient(valueWeights3[i, j]);
                double momentum = 0.9 * valueWeights3[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                valueWeights3[i, j] -= update;
                valueWeights3[i, j] = gradient;
            }
        }

        // Update value output weights
        for (int i = 0; i < valueOutputWeights.Length; i++)
        {
            double gradient = loss * CalculateOutputGradient(valueOutputWeights[i]);
            double momentum = 0.9 * valueOutputWeights[i];
            double update = currentLearningRate * (gradient + momentum);
            update = Math.Clamp(update, -1.0, 1.0);
            valueOutputWeights[i] -= update;
            valueOutputWeights[i] = gradient;
        }
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
        double entropySum = 0;

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

            if (policyLoss > 1e10 || valueLoss > 1e10)
            {
                Console.WriteLine("Loss too large, check gradients!");
            }
            // Calculate entropy for this distribution
            // Correct entropy calculation
            double Entropy = -currentProbs.Sum(p => p * Math.Log(Math.Max(p, 1e-6)));
            entropySum += Entropy; // Accumulate for batch
        }

        // Average losses
        int batchSize = batchIndices.Count;
        policyLoss /= batchSize;
        valueLoss /= batchSize;
        //Console.WriteLine($"entropy setings: {entropySum} {batchSize}");
        double entropy = entropySum / batchSize;

        // Dynamic entropy coefficient - decay over time but maintain minimum exploration
        ENTROPY_COEF = Math.Max(0.001, 0.001 * (1.0 - episodeRewards.Count * 0.00005));

        // Track metrics
        policyLosses.Add(policyLoss);
        valueLosses.Add(valueLoss);
        entropyValues.Add(entropy);

        // Apply updates - only add the entropy term if entropy is below target
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