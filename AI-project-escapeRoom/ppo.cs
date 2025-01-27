using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;

class PPO
{
    /// <summary>
    /// Policy network weights for the first layer.
    /// </summary>
    private double[,] policyWeights1;

    /// <summary>
    /// Policy network weights for the second layer.
    /// </summary>
    private double[] policyWeights2;

    /// <summary>
    /// Value network weights for the first layer.
    /// </summary>
    private double[,] valueWeights1;

    /// <summary>
    /// Value network weights for the second layer.
    /// </summary>
    private double[] valueWeights2;

    private const double Gamma = 0.99f;
    private const double ClipEpsilon = 0.2f;
    private const double LearningRate = 0.0005f;
    private const int Epochs = 4;
    private const int HiddenSize = 64;
    private const double EntropyCoef = 0.02f;

    private Random random;
    private int stateSize;


    /// <summary>
    /// Initializes a new instance of the PPO class.
    /// Sets up the neural networks for both policy and value functions.
    /// </summary>
    public PPO()
    {
        random = new Random();
        // For a 10x10 grid, relative positions can be in range [-9,9] for both row and col
        // So state space is 19 * 19 = 361
        stateSize = 361;

        // Initialize policy network weights
        policyWeights1 = InitializeWeights(stateSize, HiddenSize);
        policyWeights2 = InitializeWeights(HiddenSize, 4).Cast<double>().ToArray();

        // Initialize value network weights
        valueWeights1 = InitializeWeights(stateSize, HiddenSize);
        valueWeights2 = InitializeWeights(HiddenSize, 1).Cast<double>().ToArray();
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
    /// Computes action probabilities for a given state using the policy network.
    /// </summary>
    /// <param name="stateVector">Vector representation of the state</param>
    /// <returns>Probability distribution over possible actions</returns>
    private double[] Policy(double[] stateVector)
    {
        var hidden = ReLU(LinearLayer(stateVector, policyWeights1));
        var logits = LinearLayer(hidden, new double[HiddenSize, 5], policyWeights2);
        return Softmax(logits);
    }
    /// <summary>
    /// Estimates the value of a state using the value network.
    /// </summary>
    /// <param name="stateVector">Vector representation of the state</param>
    /// <returns>Estimated value of the state</returns>
    private double Value(double[] stateVector)
    {
        var hidden = ReLU(LinearLayer(stateVector, valueWeights1));
        var value = LinearLayer(hidden, new double[HiddenSize, 1], valueWeights2);
        return value[0];
    }

    /// <summary>
    /// Computes advantages using Generalized Advantage Estimation (GAE).
    /// </summary>
    /// <param name="rewards">List of rewards from the episode</param>
    /// <param name="values">List of estimated state values</param>
    /// <returns>List of normalized advantages</returns>
    private List<double> ComputeAdvantages(List<int> rewards, List<double> values)
    {
        List<double> advantages = new List<double>();
        double nextValue = 0;
        double advantage = 0;

        for (int t = rewards.Count - 1; t >= 0; t--)
        {
            double delta = rewards[t] + Gamma * nextValue - values[t];
            advantage = delta + Gamma * 0.95f * advantage;
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

    /// <summary>
    /// Trains the agent using the PPO algorithm.
    /// </summary>
    /// <param name="env">The environment to train in</param>
    /// <param name="episodes">Number of episodes to train for</param>
    public void Train(GameEnvironment env, int episodes)
    {
        double bestReward = double.MinValue;
        int episode = 0;

        for (; episode < episodes; episode++)
        {
            List<double[]> stateVectors = new List<double[]>();
            List<int> actions = new List<int>();
            List<int> rewards = new List<int>();
            List<double> oldActionProbs = new List<double>();

            double[] state = env.GetState();
            bool done = false;
            int totalReward = 0;

            // Collect trajectory
            while (!done)
            {
                double[] stateVector = state.Select(s => (double)s).ToArray();
                double[] actionProbs = Policy(stateVector);
                int action = SampleAction(actionProbs);
                double actionProb = actionProbs[action];

                stateVectors.Add(stateVector);
                actions.Add(action);
                oldActionProbs.Add(actionProb);

                System.Console.WriteLine("Action: " + action + " Action Prob: " + actionProb + " State: " + stateVector[0] + " " + stateVector[1] + " " + stateVector[2] + " " + stateVector[3] + " " + stateVector[4] + " " + stateVector[5] + " reward: " + totalReward);
                (state, int reward, done) = env.Step(action);
                rewards.Add(reward);
                totalReward += reward;
                Thread.Sleep(10); // Slow down visualization

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

    /// <summary>
    /// Updates both policy and value networks using collected trajectory data.
    /// </summary>
    /// <param name="states">List of state vectors</param>
    /// <param name="actions">List of actions taken</param>
    /// <param name="advantages">List of computed advantages</param>
    /// <param name="oldActionProbs">List of action probabilities from old policy</param>
    /// <param name="values">List of estimated state values</param>
    /// <param name="rewards">List of rewards received</param>
    private void UpdateNetworks(List<double[]> states, List<int> actions, List<double> advantages,
                              List<double> oldActionProbs, List<double> values, List<int> rewards)
    {
        for (int i = 0; i < states.Count; i++)
        {
            var currentProbs = Policy(states[i]);
            double ratio = currentProbs[actions[i]] / oldActionProbs[i];

            // Policy loss
            double clippedRatio = Math.Clamp(ratio, 1 - ClipEpsilon, 1 + ClipEpsilon);
            double policyLoss = -Math.Min(ratio * advantages[i], clippedRatio * advantages[i]);

            // Add entropy bonus for exploration
            double entropy = -currentProbs.Sum(p => p * Math.Log(Math.Max(p, 1e-10)));
            policyLoss -= EntropyCoef * entropy;

            // Value loss
            double returns = advantages[i] + values[i];
            double valueLoss = Math.Pow(Value(states[i]) - returns, 2);

            // Update weights using simple gradient descent
            UpdatePolicyWeights(states[i], actions[i], policyLoss);
            UpdateValueWeights(states[i], valueLoss);
        }
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
                policyWeights1[i, j] -= LearningRate * loss * state[i];

        for (int i = 0; i < policyWeights2.Length; i++)
            policyWeights2[i] -= LearningRate * loss;
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
                valueWeights1[i, j] -= LearningRate * loss * state[i];

        for (int i = 0; i < valueWeights2.Length; i++)
            valueWeights2[i] -= LearningRate * loss;
    }

}