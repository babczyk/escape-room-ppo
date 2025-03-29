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
    private static Random random = new Random();
    public int curentEpisode = 0;
    public double policyLossesfordispaly = 0;
    public double Value_Loss = 0;
    public double Entropy = 0;
    public double totalRewardInEpisode = 0;

    // Neural network architecture
    private const int HIDDEN_LAYER_1_SIZE = 128;
    private const int HIDDEN_LAYER_2_SIZE = 128;
    private const int HIDDEN_LAYER_3_SIZE = 64;

    // Network weights
    private double[,] policyWeights1;
    private double[,] policyWeights2;
    private double[,] policyWeights3;
    private double[,] policyWeightsOutput;
    private double[,] valueWeights1;
    private double[,] valueWeights2;
    private double[,] valueWeights3;
    private double[,] valueWeightsOutput;


    // Bias weights
    private double[] policyBias1;
    private double[] policyBias2;
    private double[] policyBias3;
    private double[] policyOutputBias;

    private double[] valueBias1;
    private double[] valueBias2;
    private double[] valueBias3;
    private double[] valueOutputBias;

    // Hyperparameters
    private const double GAMMA = 0.99f;
    private const double CLIP_EPSILON = 0.2f;
    private const double LEARNING_RATE = 0.002f;
    private const int EPOCHS = 8;
    private double ENTROPY_COEF = 0.05f; // Made non-constant to allow decay
    private const double VALUE_COEF = 0.3f;
    private const int BATCH_SIZE = 64;

    // Training metrics
    public List<double> episodeRewards;
    private List<double> policyLosses;
    private List<double> valueLosses;
    private List<double> entropyValues;

    // BatchNorm running statistics
    private double[][] batchNormGamma; // Scale parameters
    private double[][] batchNormBeta;  // Shift parameters
    private double[] runningMean;      // Running means for each layer
    private double[] runningVar;       // Running variances for each layer
    private const double MOMENTUM = 0.99;

    // BatchNorm parameters for value network
    private double[][] valueBatchNormGamma; // Scale parameters
    private double[][] valueBatchNormBeta;  // Shift parameters
    private double[] valueRunningMean;      // Running means for each layer
    private double[] valueRunningVar;       // Running variances for each layer

    private int stateSize;
    private int actionSize;

    private PPOHelper helper = new PPOHelper();

    /// <summary>
    /// Initializes a new instance of the PPO class, setting up neural networks.
    /// </summary>
    /// <param name="stateSize">Number of inputs in the state vector</param>
    /// <param name="actionSize">Number of possible actions</param>
    public PPO(int stateSize = 11, int actionSize = 5)
    {
        this.stateSize = stateSize;
        this.actionSize = actionSize;

        // Initialize policy network
        policyWeights1 = helper.InitializeWeights(stateSize, HIDDEN_LAYER_1_SIZE);
        policyWeights2 = helper.InitializeWeights(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
        policyWeights3 = helper.InitializeWeights(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE);
        policyWeightsOutput = helper.InitializeWeights(HIDDEN_LAYER_3_SIZE, actionSize);

        // Initialize value network
        valueWeights1 = helper.InitializeWeights(stateSize, HIDDEN_LAYER_1_SIZE);
        valueWeights2 = helper.InitializeWeights(HIDDEN_LAYER_1_SIZE, HIDDEN_LAYER_2_SIZE);
        valueWeights3 = helper.InitializeWeights(HIDDEN_LAYER_2_SIZE, HIDDEN_LAYER_3_SIZE);
        valueWeightsOutput = helper.InitializeWeights(HIDDEN_LAYER_3_SIZE, 1);

        // Bias initialization (small random values)
        policyBias1 = new double[HIDDEN_LAYER_1_SIZE].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();
        policyBias2 = new double[HIDDEN_LAYER_2_SIZE].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();
        policyBias3 = new double[HIDDEN_LAYER_3_SIZE].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();
        policyOutputBias = new double[actionSize].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();

        valueBias1 = new double[HIDDEN_LAYER_1_SIZE].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();
        valueBias2 = new double[HIDDEN_LAYER_2_SIZE].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();
        valueBias3 = new double[HIDDEN_LAYER_3_SIZE].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();
        valueOutputBias = new double[1].Select(_ => (random.NextDouble() - 0.5) * 0.1).ToArray();

        // Initialize training metrics
        episodeRewards = new List<double>();
        policyLosses = new List<double>();
        valueLosses = new List<double>();
        entropyValues = new List<double>();

        // Initialize BatchNorm statistics
        runningMean = new double[3];
        runningVar = new double[3];
        valueRunningMean = new double[3];
        valueRunningVar = new double[3];

        for (int i = 0; i < 3; i++)
        {
            runningVar[i] = 1.0;
            valueRunningVar[i] = 1.0;
        }

        // Initialize BatchNorm parameters
        batchNormGamma = new double[3][];
        batchNormBeta = new double[3][];
        valueBatchNormGamma = new double[3][];
        valueBatchNormBeta = new double[3][];

        for (int i = 0; i < 3; i++)
        {
            int layerSize = (i == 0) ? HIDDEN_LAYER_1_SIZE :
                            (i == 1) ? HIDDEN_LAYER_2_SIZE : HIDDEN_LAYER_3_SIZE;

            batchNormGamma[i] = new double[layerSize];
            batchNormBeta[i] = new double[layerSize];
            valueBatchNormGamma[i] = new double[layerSize];
            valueBatchNormBeta[i] = new double[layerSize];

            // Initialize gamma and beta
            for (int j = 0; j < layerSize; j++)
            {
                batchNormGamma[i][j] = 0.5;   // Less aggressive than 1.0
                batchNormBeta[i][j] = 0.0;   // Small shift

                valueBatchNormGamma[i][j] = 0.5;
                valueBatchNormBeta[i][j] = 0.0;
            }
        }
    }

    /// <summary>
    /// Performs a forward pass through the policy network to predict action probabilities.
    /// </summary>
    /// <param name="input">The current state as a vector</param>
    /// <param name="isTraining">Indicates if the model is in training mode</param>
    /// <returns>A probability distribution over possible actions</returns>
    private double[] PolicyForward(double[] input, bool isTraining = true)
    {
        // First layer
        var z1 = helper.LinearLayer(input, policyWeights1, policyBias1);
        var bn1 = BatchNormalize(z1, 0, isTraining); // Layer index 0
        var layer1 = helper.LeakyReLU(bn1);

        // Second layer
        var z2 = helper.LinearLayer(layer1, policyWeights2, policyBias2);
        var bn2 = BatchNormalize(z2, 1, isTraining); // Layer index 1
        var layer2 = helper.LeakyReLU(bn2);

        // Third layer
        var z3 = helper.LinearLayer(layer2, policyWeights3, policyBias3);
        var bn3 = BatchNormalize(z3, 2, isTraining); // Layer index 2
        var layer3 = helper.LeakyReLU(bn3);

        // Output layer - typically no BatchNorm on output layer
        var output = helper.LinearLayer(layer3, policyWeightsOutput, policyOutputBias);

        // You may want to reduce this scaling factor - 1000 is very high
        for (int i = 0; i < output.Length; i++)
        {
            output[i] *= 2; // More reasonable scaling
        }

        var probabilities = helper.Softmax(output);
        return probabilities;
    }


    /// <summary>
    /// Normalizes a batch of data using batch normalization.
    /// </summary>
    /// <param name="layer">The input layer values</param>
    /// <param name="layerIndex">Index of the layer being normalized</param>
    /// <param name="isTraining">Indicates if the model is in training mode</param>
    /// <returns>Normalized values for the given layer</returns>
    private double[] BatchNormalize(double[] layer, int layerIndex, bool isTraining)
    {
        int layerSize = layer.Length;
        double[] normalized = new double[layerSize];

        // Parameters for this layer (you need to add these to your class)
        double[] gamma = batchNormGamma[layerIndex]; // Scale parameter (initialized to 1s)
        double[] beta = batchNormBeta[layerIndex];   // Shift parameter (initialized to 0s)

        // Calculate batch statistics
        double mean = 0, variance = 0;

        if (isTraining)
        {
            // Compute mean
            for (int i = 0; i < layerSize; i++)
            {
                mean += layer[i];
            }
            mean /= layerSize;

            // Compute variance
            for (int i = 0; i < layerSize; i++)
            {
                double diff = layer[i] - mean;
                variance += diff * diff;
            }
            variance /= layerSize;

            // Update running statistics for inference
            runningMean[layerIndex] = MOMENTUM * runningMean[layerIndex] + (1 - MOMENTUM) * mean;
            runningVar[layerIndex] = MOMENTUM * runningVar[layerIndex] + (1 - MOMENTUM) * variance;
        }
        else
        {
            // Use running statistics during inference
            mean = runningMean[layerIndex];
            variance = runningVar[layerIndex];
        }

        // Small constant to avoid division by zero
        double epsilon = 1e-5;

        // Normalize and apply scale and shift
        for (int i = 0; i < layerSize; i++)
        {
            normalized[i] = gamma[i] * ((layer[i] - mean) / Math.Sqrt(variance + epsilon)) + beta[i];
        }

        return normalized;
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
            double std = Math.Sqrt(advantages.Select(x => Math.Pow(x - mean, 2)).Average() + 1e-5);
            return advantages.Select(a => (a - mean) / std).ToList();
        }
        return advantages;
    }

    /// <summary>
    /// Computes a forward pass through the value network to estimate state value.
    /// </summary>
    /// <param name="input">The current state as a vector</param>
    /// <param name="isTraining">Indicates if the model is in training mode</param>
    /// <returns>A single scalar value representing the estimated value of the state</returns>
    private double ValueForward(double[] input, bool isTraining = true)
    {
        // First layer
        var z1 = helper.LinearLayer(input, valueWeights1, valueBias1);
        var bn1 = BatchNormalizeValue(z1, 0, isTraining); // Layer index 0
        var layer1 = helper.LeakyReLU(bn1);

        // Second layer
        var z2 = helper.LinearLayer(layer1, valueWeights2, valueBias2);
        var bn2 = BatchNormalizeValue(z2, 1, isTraining); // Layer index 1
        var layer2 = helper.LeakyReLU(bn2);

        // Third layer
        var z3 = helper.LinearLayer(layer2, valueWeights3, valueBias3);
        var bn3 = BatchNormalizeValue(z3, 2, isTraining); // Layer index 2
        var layer3 = helper.LeakyReLU(bn3);

        // Output layer - typically no BatchNorm on output layer for value function
        var output = helper.LinearLayer(layer3, valueWeightsOutput, valueOutputBias);
        return output[0];  // Single value output
    }


    /// <summary>
    /// Normalizes a batch of data using batch normalization.
    /// </summary>
    /// <param name="layer">The input layer values</param>
    /// <param name="layerIndex">Index of the layer being normalized</param>
    /// <param name="isTraining">Indicates if the model is in training mode</param>
    /// <returns>Normalized values for the given layer</returns>
    private double[] BatchNormalizeValue(double[] layer, int layerIndex, bool isTraining)
    {
        int layerSize = layer.Length;
        double[] normalized = new double[layerSize];

        // Parameters for this layer
        double[] gamma = valueBatchNormGamma[layerIndex]; // Scale parameter
        double[] beta = valueBatchNormBeta[layerIndex];   // Shift parameter

        // Calculate batch statistics
        double mean = 0, variance = 0;

        if (isTraining)
        {
            // Compute mean
            for (int i = 0; i < layerSize; i++)
            {
                mean += layer[i];
            }
            mean /= layerSize;

            // Compute variance
            for (int i = 0; i < layerSize; i++)
            {
                double diff = layer[i] - mean;
                variance += diff * diff;
            }
            variance /= layerSize;

            // Update running statistics for inference
            valueRunningMean[layerIndex] = MOMENTUM * valueRunningMean[layerIndex] + (1 - MOMENTUM) * mean;
            valueRunningVar[layerIndex] = MOMENTUM * valueRunningVar[layerIndex] + (1 - MOMENTUM) * variance;
        }
        else
        {
            // Use running statistics during inference
            mean = valueRunningMean[layerIndex];
            variance = valueRunningVar[layerIndex];
        }

        // Small constant to avoid division by zero
        double epsilon = 1e-5;

        // Normalize and apply scale and shift
        for (int i = 0; i < layerSize; i++)
        {
            normalized[i] = gamma[i] * ((layer[i] - mean) / Math.Sqrt(variance + epsilon)) + beta[i];
        }

        return normalized;
    }

    /// <summary>
    /// Collects a trajectory from the environment by running an episode.
    /// </summary>
    /// <param name="env">The game environment</param>
    /// <returns>A trajectory object containing states, actions, rewards, and total episode reward</returns>
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
            int action = helper.SampleAction(actionProbs);

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

    /// <summary>
    /// Trains the PPO model using collected data over multiple episodes.
    /// </summary>
    /// <param name="env">The game environment</param>
    /// <param name="episodes">Number of training episodes</param>
    /// <param name="modelPath">File path to save the trained model</param>
    /// <param name="progressPath">File path to save training progress</param>
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
        }
        for (; episode < episodes; episode++)
        {
            curentEpisode = episode;
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
            totalRewardInEpisode = totalReward;
            averageReward = episodeRewards.TakeLast(5).Average(); //take an avrege of 5 episodes (5 generations)
            // Log progress
            if (episode % 5 == 0)
            {
                policyLossesfordispaly = policyLosses.LastOrDefault();
                Value_Loss = valueLosses.LastOrDefault();
                Entropy = entropyValues.LastOrDefault();
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
        }
    }

    /// <summary>
    /// Loads a model from a JSON file.
    /// </summary>
    /// <typeparam name="T"></typeparam>
    /// <param name="filePath"></param>
    /// <returns></returns>
    // Load the entire model including weights, biases, and batch norm stats
    private void LoadModel(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"Model file not found at {filePath}");

        try
        {
            string json = File.ReadAllText(filePath);

            using (JsonDocument document = JsonDocument.Parse(json))
            {
                var root = document.RootElement;

                policyWeights1 = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "POLICY1"));
                // Load policy network weights
                policyWeights2 = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "POLICY2"));
                policyWeights3 = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "POLICY3"));
                policyWeightsOutput = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "POLICY_WEIGHTS_OUTPUT"));
                policyBias1 = helper.SafeDeserialize1D(root, "POLICY_BIAS1");
                policyBias2 = helper.SafeDeserialize1D(root, "POLICY_BIAS2");
                policyBias3 = helper.SafeDeserialize1D(root, "POLICY_BIAS3");
                policyOutputBias = helper.SafeDeserialize1D(root, "POLICY_OUTPUT_BIAS");

                // Load value network weights
                valueWeights1 = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "VALUE1"));
                valueWeights2 = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "VALUE2"));
                valueWeights3 = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "VALUE3"));
                valueWeightsOutput = helper.ConvertTo2DArray(helper.SafeDeserialize2D(root, "VALUE_WEIGHTS_OUTPUT"));

                // Load BatchNorm statistics
                runningMean = helper.SafeDeserialize1D(root, "VALUE_RUNNING_MEAN");
                runningVar = helper.SafeDeserialize1D(root, "VALUE_RUNNING_VAR");
                Console.WriteLine($"Running Mean: {string.Join(", ", runningMean)}");
            }

            Console.WriteLine($"Model successfully loaded from {filePath}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error loading model: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// Loads the policy and value networks from a JSON file.
    /// </summary>
    /// <param name="filePath"></param>
    /// <returns></returns>
    /// <exception cref="InvalidOperationException"></exception>
    public (int episode, double bestReward, List<double> recentRewards) LoadProgress(string filePath)
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
                                    .Select(e => e.GetDouble())
                                    .ToList();

        return (episodeA, bestReward, recentRewards);
    }

    /// <summary>
    /// 
    /// </summary>
    /// <param name="filePath"></param>
    /// <param name="model"></param>
    // Save the entire model including weights, biases, and batch norm stats
    private void SaveModel(string filePath, int episode = 1)
    {
        var model = new
        {
            IN_EPISODE = episode,

            // Policy network weights and biases
            POLICY1 = helper.ConvertToJaggedArray(policyWeights1),
            POLICY2 = helper.ConvertToJaggedArray(policyWeights2),
            POLICY3 = helper.ConvertToJaggedArray(policyWeights3),
            POLICY_WEIGHTS_OUTPUT = helper.ConvertToJaggedArray(policyWeightsOutput),
            POLICY_BIAS1 = policyBias1,
            POLICY_BIAS2 = policyBias2,
            POLICY_BIAS3 = policyBias3,
            POLICY_OUTPUT_BIAS = policyOutputBias,

            // Value network weights
            VALUE1 = helper.ConvertToJaggedArray(valueWeights1),
            VALUE2 = helper.ConvertToJaggedArray(valueWeights2),
            VALUE3 = helper.ConvertToJaggedArray(valueWeights3),
            VALUE_WEIGHTS_OUTPUT = helper.ConvertToJaggedArray(valueWeightsOutput),

            // BatchNorm statistics
            VALUE_BIAS1 = valueBias1,
            VALUE_BIAS2 = valueBias2,
            VALUE_BIAS3 = valueBias3,
            VALUE_OUTPUT_BIAS = valueOutputBias,

            // Add batch norm parameters
            POLICY_BATCH_NORM_GAMMA = batchNormGamma,
            POLICY_BATCH_NORM_BETA = batchNormBeta,
            VALUE_BATCH_NORM_GAMMA = valueBatchNormGamma,
            VALUE_BATCH_NORM_BETA = valueBatchNormBeta,

            // Add value network running stats
            VALUE_RUNNING_MEAN = valueRunningMean,
            VALUE_RUNNING_VAR = valueRunningVar
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


    private void UpdateNetworkWeights(double totalLoss)
    {
        // Calculate gradients and update policy network weights
        UpdatePolicyNetworkWeights(totalLoss);

        // Calculate gradients and update value network weights
        UpdateValueNetworkWeights(totalLoss);

        //update bias
        UpdateBiases(totalLoss);

    }
    private void UpdateBiases(double loss)
    {
        // Learning rate with decay
        double currentLearningRate = LEARNING_RATE * (1.0 / (1.0 + 0.0001 * episodeRewards.Count));

        // Update policy network layer 1
        for (int i = 0; i < policyBias1.Length; i++)
        {
            // Gradient descent update with momentum
            double gradient = loss * helper.CalculateLayerGradient(policyBias1[i]);
            double momentum = 0.9 * policyBias1[i];
            double update = currentLearningRate * (gradient + momentum);

            // Update weight with gradient clipping
            update = Math.Clamp(update, -1.0, 1.0);
            policyBias1[i] -= update;

            // Store gradient for momentum
            //policyWeights1[i, j] = gradient;
        }

        // Update policy network layer 2
        for (int i = 0; i < policyBias2.Length; i++)
        {
            double gradient = loss * helper.CalculateLayerGradient(policyBias2[i]);
            double momentum = 0.9 * policyBias2[i];
            double update = currentLearningRate * (gradient + momentum);
            update = Math.Clamp(update, -1.0, 1.0);
            policyBias2[i] -= update;
            //policyWeights2[i, j] = gradient;
        }

        // Update policy network layer 3
        for (int i = 0; i < policyBias3.Length; i++)
        {
            double gradient = loss * helper.CalculateLayerGradient(policyBias3[i]);
            double momentum = 0.9 * policyBias3[i];
            double update = currentLearningRate * (gradient + momentum);
            update = Math.Clamp(update, -1.0, 1.0);
            policyBias3[i] -= update;
            //policyWeights3[i, j] = gradient;
        }

        // Update policy output weights
        for (int i = 0; i < policyOutputBias.Length; i++)
        {
            double gradient = loss * helper.CalculateOutputGradient(policyOutputBias[i]);
            double momentum = 0.9 * policyOutputBias[i];
            double update = currentLearningRate * (gradient + momentum);
            update = Math.Clamp(update, -1.0, 1.0);
            policyOutputBias[i] -= update;
            //policyOutputWeights[i] = gradient;
        }
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
                double gradient = loss * helper.CalculateLayerGradient(policyWeights1[i, j]);
                double momentum = 0.9 * policyWeights1[i, j];
                double update = currentLearningRate * (gradient + momentum);

                // Update weight with gradient clipping
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeights1[i, j] -= update;

                // Store gradient for momentum
                //policyWeights1[i, j] = gradient;
            }
        }

        // Update policy network layer 2
        for (int i = 0; i < policyWeights2.GetLength(0); i++)
        {
            for (int j = 0; j < policyWeights2.GetLength(1); j++)
            {
                double gradient = loss * helper.CalculateLayerGradient(policyWeights2[i, j]);
                double momentum = 0.9 * policyWeights2[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeights2[i, j] -= update;
                //policyWeights2[i, j] = gradient;
            }
        }

        // Update policy network layer 3
        for (int i = 0; i < policyWeights3.GetLength(0); i++)
        {
            for (int j = 0; j < policyWeights3.GetLength(1); j++)
            {
                double gradient = loss * helper.CalculateLayerGradient(policyWeights3[i, j]);
                double momentum = 0.9 * policyWeights3[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeights3[i, j] -= update;
                //policyWeights3[i, j] = gradient;
            }
        }

        // Update policy output weights
        for (int i = 0; i < policyWeightsOutput.GetLength(0); i++)
        {
            for (int j = 0; j < policyWeightsOutput.GetLength(1); j++)
            {
                double gradient = loss * helper.CalculateOutputGradient(policyWeightsOutput[i, j]);
                double momentum = 0.9 * policyWeightsOutput[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                policyWeightsOutput[i, j] -= update;
            }
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
                double gradient = loss * helper.CalculateLayerGradient(valueWeights1[i, j]);
                double momentum = 0.9 * valueWeights1[i, j];
                double update = currentLearningRate * (gradient + momentum);

                // Apply gradient clipping
                update = Math.Clamp(update, -1.0, 1.0);

                // Update weights
                valueWeights1[i, j] -= update;

                // Store gradient for future momentum updates
                //valueWeights1[i, j] = gradient;
            }
        }

        // Update value network layer 2
        for (int i = 0; i < valueWeights2.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeights2.GetLength(1); j++)
            {
                double gradient = loss * helper.CalculateLayerGradient(valueWeights2[i, j]);
                double momentum = 0.9 * valueWeights2[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                valueWeights2[i, j] -= update;
                //valueWeights2[i, j] = gradient;
            }
        }

        // Update value network layer 3
        for (int i = 0; i < valueWeights3.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeights3.GetLength(1); j++)
            {
                double gradient = loss * helper.CalculateLayerGradient(valueWeights3[i, j]);
                double momentum = 0.9 * valueWeights3[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                valueWeights3[i, j] -= update;
                //valueWeights3[i, j] = gradient;
            }
        }

        // Update value output weights
        for (int i = 0; i < valueWeightsOutput.GetLength(0); i++)
        {
            for (int j = 0; j < valueWeightsOutput.GetLength(1); j++)
            {
                double gradient = loss * helper.CalculateOutputGradient(valueWeightsOutput[i, j]);
                double momentum = 0.9 * valueWeightsOutput[i, j];
                double update = currentLearningRate * (gradient + momentum);
                update = Math.Clamp(update, -1.0, 1.0);
                valueWeightsOutput[i, j] -= update;
            }
        }
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
            double ratio = currentProbs[trajectory.actions[idx]];

            // Policy loss with clipping
            double advantage = trajectory.advantages[idx];
            double unclippedLoss = ratio * advantage;
            double clippedLoss = Math.Clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantage;
            policyLoss = -Math.Min(unclippedLoss, clippedLoss);

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
        double entropy = entropySum / batchSize;

        // Dynamic entropy coefficient - decay over time but maintain minimum exploration
        ENTROPY_COEF = Math.Max(0.0001, ENTROPY_COEF * 0.995);

        // Track metrics
        policyLosses.Add(policyLoss);
        valueLosses.Add(valueLoss);
        entropyValues.Add(entropy);

        // Apply updates - only add the entropy term if entropy is below target
        double totalLoss = policyLoss + VALUE_COEF * valueLoss + ENTROPY_COEF * entropy;
        UpdateNetworkWeights(totalLoss);
    }
}