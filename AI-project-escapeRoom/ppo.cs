using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using AI_project_escapeRoom;
using MathNet.Numerics.LinearAlgebra;

namespace PPOReinforcementLearning
{
    /// <summary>
    /// Represents a single experience step for the agent
    /// </summary>
    public class Experience
    {
        public Vector<float> State { get; set; }
        public int Action { get; set; }
        public float Reward { get; set; }
        public Vector<float> NextState { get; set; }
        public bool Done { get; set; }
        public float LogProbability { get; set; }
        public float Value { get; set; }
    }

    /// <summary>
    /// Neural network for the actor and critic models
    /// </summary>
    public class NeuralNetwork
    {
        private List<Matrix<float>> weights = new List<Matrix<float>>();
        private List<Vector<float>> biases = new List<Vector<float>>();
        private List<int> layerSizes;
        private Random random = new Random();

        public NeuralNetwork(List<int> layerSizes)
        {
            this.layerSizes = layerSizes;
            InitializeParameters();
        }

        private void InitializeParameters()
        {
            for (int i = 0; i < layerSizes.Count - 1; i++)
            {
                int inputSize = layerSizes[i];
                int outputSize = layerSizes[i + 1];

                Matrix<float> weight = CreateMatrix(outputSize, inputSize);
                Vector<float> bias = CreateVector(outputSize);

                weights.Add(weight);
                biases.Add(bias);
            }
        }

        private Matrix<float> CreateMatrix(int rows, int cols)
        {
            var matrix = Matrix<float>.Build.Dense(rows, cols);

            // Xavier initialization
            float scale = (float)Math.Sqrt(6.0 / (rows + cols));
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    matrix[i, j] = (float)((random.NextDouble() * 2 - 1) * scale);
                }
            }

            return matrix;
        }

        private Vector<float> CreateVector(int size)
        {
            return Vector<float>.Build.Dense(size);
        }

        public Vector<float> Forward(Vector<float> input)
        {
            Vector<float> activation = input;

            for (int i = 0; i < weights.Count - 1; i++)
            {
                activation = weights[i].Multiply(activation).Add(biases[i]);
                activation = ApplyReLU(activation);
            }

            // Output layer
            activation = weights[weights.Count - 1].Multiply(activation).Add(biases[weights.Count - 1]);

            return activation;
        }

        private Vector<float> ApplyReLU(Vector<float> input)
        {
            return Vector<float>.Build.DenseOfEnumerable(
                input.Select(x => Math.Max(0, x))
            );
        }

        public List<Matrix<float>> GetWeights() => weights;

        public List<Vector<float>> GetBiases() => biases;

        public void SetWeights(List<Matrix<float>> newWeights) => weights = newWeights;

        public void SetBiases(List<Vector<float>> newBiases) => biases = newBiases;
    }

    /// <summary>
    /// Adam optimizer implementation
    /// </summary>
    public class AdamOptimizer
    {
        private float learningRate;
        private float beta1;
        private float beta2;
        private float epsilon;
        private List<Matrix<float>> mWeights;
        private List<Vector<float>> mBiases;
        private List<Matrix<float>> vWeights;
        private List<Vector<float>> vBiases;
        private int t = 0;

        public AdamOptimizer(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1e-8f)
        {
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.epsilon = epsilon;
        }

        public void Initialize(List<Matrix<float>> weights, List<Vector<float>> biases)
        {
            mWeights = new List<Matrix<float>>();
            mBiases = new List<Vector<float>>();
            vWeights = new List<Matrix<float>>();
            vBiases = new List<Vector<float>>();

            foreach (var weight in weights)
            {
                mWeights.Add(Matrix<float>.Build.Dense(weight.RowCount, weight.ColumnCount));
                vWeights.Add(Matrix<float>.Build.Dense(weight.RowCount, weight.ColumnCount));
            }

            foreach (var bias in biases)
            {
                mBiases.Add(Vector<float>.Build.Dense(bias.Count));
                vBiases.Add(Vector<float>.Build.Dense(bias.Count));
            }
        }

        public void Update(
            List<Matrix<float>> weights,
            List<Vector<float>> biases,
            List<Matrix<float>> gradWeights,
            List<Vector<float>> gradBiases)
        {
            t++;
            float correctedLR = learningRate * (float)Math.Sqrt(1 - Math.Pow(beta2, t)) / (1 - (float)Math.Pow(beta1, t));

            for (int i = 0; i < weights.Count; i++)
            {
                // Update weight momentum and RMS
                mWeights[i] = mWeights[i].Multiply(beta1).Add(gradWeights[i].Multiply(1 - beta1));
                vWeights[i] = vWeights[i].Multiply(beta2).Add(gradWeights[i].PointwiseMultiply(gradWeights[i]).Multiply(1 - beta2));

                // Calculate update
                Matrix<float> update = CalculateUpdate(mWeights[i], vWeights[i], correctedLR);
                weights[i] = weights[i].Subtract(update);

                // Update bias momentum and RMS
                mBiases[i] = mBiases[i].Multiply(beta1).Add(gradBiases[i].Multiply(1 - beta1));
                vBiases[i] = vBiases[i].Multiply(beta2).Add(gradBiases[i].PointwiseMultiply(gradBiases[i]).Multiply(1 - beta2));

                // Calculate bias update
                Vector<float> biasUpdate = CalculateUpdate(mBiases[i], vBiases[i], correctedLR);
                biases[i] = biases[i].Subtract(biasUpdate);
            }
        }

        private Matrix<float> CalculateUpdate(Matrix<float> m, Matrix<float> v, float correctedLR)
        {
            var update = Matrix<float>.Build.Dense(m.RowCount, m.ColumnCount);

            for (int i = 0; i < m.RowCount; i++)
            {
                for (int j = 0; j < m.ColumnCount; j++)
                {
                    update[i, j] = correctedLR * m[i, j] / ((float)Math.Sqrt(v[i, j]) + epsilon);
                }
            }

            return update;
        }

        private Vector<float> CalculateUpdate(Vector<float> m, Vector<float> v, float correctedLR)
        {
            var update = Vector<float>.Build.Dense(m.Count);

            for (int i = 0; i < m.Count; i++)
            {
                update[i] = correctedLR * m[i] / ((float)Math.Sqrt(v[i]) + epsilon);
            }

            return update;
        }
    }

    /// <summary>
    /// PPO Agent implementation
    /// </summary>
    public class PPOAgent
    {
        public NeuralNetwork actorNetwork;
        public NeuralNetwork criticNetwork;
        public AdamOptimizer actorOptimizer;
        public AdamOptimizer criticOptimizer;
        private int stateSize;
        private int actionSize;
        private float gamma = 0.99f;
        private float clipEpsilon = 0.2f;
        private float valueCoeff = 0.5f;
        private float entropyCoeff = 0.01f;
        private int batchSize = 64;
        private int epochs = 10;
        private Random random = new Random();

        public PPOAgent(int stateSize, int actionSize)
        {
            this.stateSize = stateSize;
            this.actionSize = actionSize;

            // Initialize actor network (policy)
            actorNetwork = new NeuralNetwork(new List<int> { stateSize, 64, 64, actionSize });

            // Initialize critic network (value function)
            criticNetwork = new NeuralNetwork(new List<int> { stateSize, 64, 64, 1 });

            // Initialize optimizers
            actorOptimizer = new AdamOptimizer();
            criticOptimizer = new AdamOptimizer();

            actorOptimizer.Initialize(actorNetwork.GetWeights(), actorNetwork.GetBiases());
            criticOptimizer.Initialize(criticNetwork.GetWeights(), criticNetwork.GetBiases());
        }

        public int ChooseAction(Vector<float> state)
        {
            Vector<float> logits = actorNetwork.Forward(state);
            Vector<float> probs = Softmax(logits);

            // Sample action from probability distribution
            float sample = (float)random.NextDouble();
            float cumulativeProbability = 0;

            for (int i = 0; i < actionSize; i++)
            {
                cumulativeProbability += probs[i];
                if (sample <= cumulativeProbability)
                {
                    return i;
                }
            }

            return actionSize - 1;
        }

        public float GetLogProbability(Vector<float> state, int action)
        {
            Vector<float> logits = actorNetwork.Forward(state);
            Vector<float> probs = Softmax(logits);
            return (float)Math.Log(probs[action] + 1e-10);
        }

        public float GetValue(Vector<float> state)
        {
            Vector<float> value = criticNetwork.Forward(state);
            return value[0];
        }

        private Vector<float> Softmax(Vector<float> logits)
        {
            Vector<float> expValues = Vector<float>.Build.DenseOfEnumerable(
                logits.Select(x => (float)Math.Exp(x - logits.Max()))
            );

            float sumExpValues = expValues.Sum();
            return expValues.Divide(sumExpValues);
        }

        public void Train(List<Experience> experiences)
        {
            // Calculate returns and advantages
            List<float> returns = new List<float>();
            List<float> advantages = new List<float>();
            ComputeReturnsAndAdvantages(experiences, returns, advantages);

            // Train for multiple epochs
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                // Create mini-batches
                List<int> indices = Enumerable.Range(0, experiences.Count).ToList();
                Shuffle(indices);

                for (int i = 0; i < experiences.Count; i += batchSize)
                {
                    List<int> batchIndices = indices.Skip(i).Take(batchSize).ToList();
                    UpdateNetworks(experiences, returns, advantages, batchIndices);
                }
            }
        }

        private void ComputeReturnsAndAdvantages(List<Experience> experiences, List<float> returns, List<float> advantages)
        {
            float nextValue = 0;

            for (int i = experiences.Count - 1; i >= 0; i--)
            {
                Experience exp = experiences[i];
                float nextReturn = exp.Done ? 0 : nextValue;
                float currentReturn = exp.Reward + gamma * nextReturn;

                returns.Insert(0, currentReturn);
                advantages.Insert(0, currentReturn - exp.Value);

                nextValue = exp.Value;
            }

            // Normalize advantages
            float meanAdvantage = advantages.Average();
            float stdAdvantage = (float)Math.Sqrt(advantages.Select(a => Math.Pow(a - meanAdvantage, 2)).Average());

            for (int i = 0; i < advantages.Count; i++)
            {
                advantages[i] = (advantages[i] - meanAdvantage) / (stdAdvantage + 1e-8f);
            }
        }

        private void UpdateNetworks(List<Experience> experiences, List<float> returns, List<float> advantages, List<int> batchIndices)
        {
            // Calculate gradients for actor and critic networks
            List<Matrix<float>> actorGradWeights = InitializeGradients(actorNetwork.GetWeights());
            List<Vector<float>> actorGradBiases = InitializeGradients(actorNetwork.GetBiases());

            List<Matrix<float>> criticGradWeights = InitializeGradients(criticNetwork.GetWeights());
            List<Vector<float>> criticGradBiases = InitializeGradients(criticNetwork.GetBiases());

            float actorLoss = 0;
            float criticLoss = 0;

            foreach (int idx in batchIndices)
            {
                Experience exp = experiences[idx];
                float advantage = advantages[idx];
                float target = returns[idx];

                // Compute policy loss
                Vector<float> logits = actorNetwork.Forward(exp.State);
                Vector<float> probs = Softmax(logits);
                float newLogProb = (float)Math.Log(probs[exp.Action] + 1e-10);

                float ratio = (float)Math.Exp(newLogProb - exp.LogProbability);
                float surrogate1 = ratio * advantage;
                float surrogate2 = Math.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantage;

                float policyLoss = -Math.Min(surrogate1, surrogate2);

                // Compute entropy bonus
                float entropy = -probs.Sum(p => p * (float)Math.Log(p + 1e-10));

                // Compute value loss
                float value = criticNetwork.Forward(exp.State)[0];
                float valueLoss = (float)Math.Pow(value - target, 2);

                // Accumulate loss
                actorLoss += policyLoss - entropyCoeff * entropy;
                criticLoss += valueCoeff * valueLoss;

                // Calculate gradients (simplified for this implementation)
                // In a real implementation, you would use automatic differentiation
                // This is a placeholder for backpropagation
                CalculateGradients(exp, advantage, target, actorGradWeights, actorGradBiases, criticGradWeights, criticGradBiases);
            }

            // Normalize gradients
            float batchSize = batchIndices.Count;
            NormalizeGradients(actorGradWeights, actorGradBiases, batchSize);
            NormalizeGradients(criticGradWeights, criticGradBiases, batchSize);

            // Update networks
            actorOptimizer.Update(actorNetwork.GetWeights(), actorNetwork.GetBiases(), actorGradWeights, actorGradBiases);
            criticOptimizer.Update(criticNetwork.GetWeights(), criticNetwork.GetBiases(), criticGradWeights, criticGradBiases);
        }

        private List<Matrix<float>> InitializeGradients(List<Matrix<float>> weights)
        {
            List<Matrix<float>> gradients = new List<Matrix<float>>();

            foreach (var weight in weights)
            {
                gradients.Add(Matrix<float>.Build.Dense(weight.RowCount, weight.ColumnCount));
            }

            return gradients;
        }

        private List<Vector<float>> InitializeGradients(List<Vector<float>> biases)
        {
            List<Vector<float>> gradients = new List<Vector<float>>();

            foreach (var bias in biases)
            {
                gradients.Add(Vector<float>.Build.Dense(bias.Count));
            }

            return gradients;
        }

        private void CalculateGradients(
            Experience exp,
            float advantage,
            float target,
            List<Matrix<float>> actorGradWeights,
            List<Vector<float>> actorGradBiases,
            List<Matrix<float>> criticGradWeights,
            List<Vector<float>> criticGradBiases)
        {
            // In a real implementation, you would compute proper gradients through backpropagation
            // This is a simplified placeholder that simulates gradient computation
            // For a real implementation, consider using a library with automatic differentiation

            // Simplified update for demonstration purposes
            // In practice, you would compute proper gradients based on the loss function
            Vector<float> actorOutput = actorNetwork.Forward(exp.State);
            Vector<float> probs = Softmax(actorOutput);
            Vector<float> actorGrad = Vector<float>.Build.Dense(probs.Count);
            actorGrad[exp.Action] = advantage;

            Vector<float> criticOutput = criticNetwork.Forward(exp.State);
            float valueDiff = criticOutput[0] - target;
            Vector<float> criticGrad = Vector<float>.Build.Dense(1);
            criticGrad[0] = valueCoeff * valueDiff;

            // Apply gradients (simplified)
            ApplySimplifiedGradients(exp.State, actorGrad, actorGradWeights, actorGradBiases);
            ApplySimplifiedGradients(exp.State, criticGrad, criticGradWeights, criticGradBiases);
        }

        private void ApplySimplifiedGradients(
            Vector<float> input,
            Vector<float> outputGrad,
            List<Matrix<float>> gradWeights,
            List<Vector<float>> gradBiases)
        {
            // Simplified gradient application for demonstration
            // In practice, you would compute proper gradients through backpropagation

            for (int layer = 0; layer < gradWeights.Count; layer++)
            {
                for (int i = 0; i < gradWeights[layer].RowCount; i++)
                {
                    for (int j = 0; j < gradWeights[layer].ColumnCount; j++)
                    {
                        gradWeights[layer][i, j] += outputGrad[i % outputGrad.Count] * input[j % input.Count] * 0.01f;
                    }

                    gradBiases[layer][i] += outputGrad[i % outputGrad.Count] * 0.01f;
                }
            }
        }

        private void NormalizeGradients(List<Matrix<float>> gradWeights, List<Vector<float>> gradBiases, float divisor)
        {
            foreach (var grad in gradWeights)
            {
                for (int i = 0; i < grad.RowCount; i++)
                {
                    for (int j = 0; j < grad.ColumnCount; j++)
                    {
                        grad[i, j] /= divisor;
                    }
                }
            }

            foreach (var grad in gradBiases)
            {
                for (int i = 0; i < grad.Count; i++)
                {
                    grad[i] /= divisor;
                }
            }
        }

        private void Shuffle<T>(List<T> list)
        {
            int n = list.Count;
            while (n > 1)
            {
                n--;
                int k = random.Next(n + 1);
                T value = list[k];
                list[k] = list[n];
                list[n] = value;
            }
        }
    }

    /// <summary>
    /// Main class for running PPO reinforcement learning
    /// </summary>
    public class PPOTrainer
    {
        private Game1 game;
        private PPOAgent agent;
        private int maxEpisodes;
        private int stepsPerEpisode;
        private int trainInterval;
        private GameEnvironment gameEnv;
        private List<float> totalReward = new List<float>();

        public PPOTrainer(Game1 game, int stateSize, int actionSize, int maxEpisodes = 1000, int stepsPerEpisode = 2000, int trainInterval = 2000)
        {
            this.game = game;
            this.maxEpisodes = maxEpisodes;
            this.stepsPerEpisode = stepsPerEpisode;
            this.trainInterval = trainInterval;
            gameEnv = new GameEnvironment(game);
            agent = new PPOAgent(stateSize, actionSize);
        }

        public async Task Train()
        {
            int totalSteps = 0;
            List<Experience> experiences = new List<Experience>();

            for (int episode = 0; episode < maxEpisodes; episode++)
            {
                Vector<float> state = gameEnv.GetState();
                float episodeReward = 0;

                for (int step = 0; step < stepsPerEpisode; step++)
                {
                    // Choose action
                    int action = agent.ChooseAction(state);

                    // Get log probability and value
                    float logProb = agent.GetLogProbability(state, action);
                    float value = agent.GetValue(state);

                    // Take action in environment
                    Console.WriteLine($"Action: {action}");
                    var (nextState, reward, done) = gameEnv.Step(action);

                    // Store experience
                    experiences.Add(new Experience
                    {
                        State = state,
                        Action = action,
                        Reward = reward,
                        NextState = nextState,
                        Done = done,
                        LogProbability = logProb,
                        Value = value
                    });

                    state = nextState;
                    episodeReward += reward;
                    totalSteps++;

                    // Check if training is needed
                    if (totalSteps % trainInterval == 0 || done)
                    {
                        // Train agent
                        agent.Train(experiences);
                        totalReward.Add(episodeReward);
                        experiences.Clear();
                    }

                    if (done)
                    {
                        break;
                    }
                }

                Console.WriteLine($"Episode {episode + 1}/{maxEpisodes}, Reward: {episodeReward}");

                // Optionally save the model periodically
                if ((episode + 1) % 1 == 0)
                {
                    SaveModel(agent, $"ppo_model_episode.json", episode + 1);
                    Console.WriteLine($"Model saved at episode {episode + 1}");
                }

                // Allow UI thread to process
                await Task.Delay(1);
            }
        }
        private void SaveModel(PPOAgent agent, string filePath, int episode = 1)
        {
            PPOHelper helper = new PPOHelper();
            var model = new
            {
                IN_EPISODE = episode,
                TOTAL_Reward = totalReward,
                IN_ACTOR_WEIGHTS = helper.ConvertToJaggedList(agent.actorNetwork.GetWeights()),  // For weights
                IN_ACTOR_BIASES = helper.ConvertVectorsToJaggedList(agent.actorNetwork.GetBiases()), // For biases

                IN_CRITIC_WEIGHTS = helper.ConvertToJaggedList(agent.criticNetwork.GetWeights()),
                IN_CRITIC_BIASES = helper.ConvertVectorsToJaggedList(agent.criticNetwork.GetBiases())


            };

            string json = JsonSerializer.Serialize(model, new JsonSerializerOptions { WriteIndented = true });
            string parentDirectory = Directory.GetParent(Directory.GetParent(Directory.
            GetParent(Directory.GetCurrentDirectory()).FullName).FullName).FullName;
            string fullPath = Path.Combine(parentDirectory + "\\statistics_R_files", filePath);
            File.WriteAllText(fullPath, json);
        }
    }





    /// <summary>
    /// Interface for environment
    /// </summary>
    public interface IEnvironment
    {
        Vector<float> Reset();
        (Vector<float> nextState, float reward, bool done) Step(int action);
    }

    /// <summary>
    /// Example implementation of IEnvironment to be replaced with actual environment
    /// </summary>
    public class YourEnvironment : IEnvironment
    {
        private int stateSize;
        private int actionSize;
        private Random random = new Random();
        private Vector<float> currentState;

        public YourEnvironment(int stateSize, int actionSize)
        {
            this.stateSize = stateSize;
            this.actionSize = actionSize;
        }

        public Vector<float> Reset()
        {
            // Initialize state
            currentState = Vector<float>.Build.Dense(stateSize, i => (float)random.NextDouble());
            return currentState;
        }

        public (Vector<float> nextState, float reward, bool done) Step(int action)
        {
            // Implement your environment dynamics here
            // This is a placeholder - replace with your actual environment logic

            // Update state based on action
            Vector<float> nextState = Vector<float>.Build.Dense(stateSize);
            for (int i = 0; i < stateSize; i++)
            {
                nextState[i] = currentState[i] + (float)(random.NextDouble() * 0.1 - 0.05);
                // Keep within bounds
                nextState[i] = Math.Max(0, Math.Min(1, nextState[i]));
            }

            // Calculate reward (placeholder)
            float reward = (float)(random.NextDouble() - 0.5);

            // Determine if episode is done
            bool done = random.NextDouble() < 0.01; // 1% chance of ending per step

            currentState = nextState;
            return (nextState, reward, done);
        }
    }

    /// <summary>
    /// Main program class
    /// </summary>
    public class Program2
    {
        public static async Task Main(Game1 game)
        {

            if (game == null)
            {
                throw new ArgumentNullException(nameof(game), "Game1 instance is null!");
            }

            // Configuration
            int stateSize = 8;    // Adjust to match your environment
            int actionSize = 4;   // Adjust to match your environment
            int maxEpisodes = 500;
            int stepsPerEpisode = 2000; // As per your requirement

            // Create environment (replace with your environment)
            // Create and run PPO trainer
            PPOTrainer trainer = new(
                game,
                stateSize,
                actionSize,
                maxEpisodes,
                stepsPerEpisode
            );

            Console.WriteLine("Starting PPO training...");
            await trainer.Train();
            Console.WriteLine("Training complete!");
        }
    }
}