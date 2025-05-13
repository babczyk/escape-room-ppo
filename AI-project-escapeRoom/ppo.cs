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
        private float gaeLambda = 0.95f;
        private float clipEpsilon = 0.15f;
        private float valueCoeff = 0.7f;
        private float entropyCoeff = 0.5f;
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
            Console.WriteLine($"State: {string.Join(", ", state)}");
            Vector<float> logits = actorNetwork.Forward(state);
            Console.WriteLine($"Logits: {string.Join(", ", logits)}");
            Vector<float> probs = Softmax(logits);
            Console.WriteLine($"Probabilities: {string.Join(", ", probs)}");
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
            float maxLogit = logits.Max();
            Vector<float> expValues = Vector<float>.Build.DenseOfEnumerable(
                logits.Select(x => (float)Math.Exp(x - maxLogit))
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
            float gae = 0f;
            float nextValue = 0f;

            for (int i = experiences.Count - 1; i >= 0; i--)
            {
                Experience exp = experiences[i];
                float gammaH = exp.Done ? 0f : gamma;
                float delta = exp.Reward + (gammaH * nextValue) - exp.Value;
                float gaeH = exp.Done ? 0f : gae;
                gae = delta + gamma * gaeLambda * gaeH;
                advantages.Insert(0, gae);

                float returnValue = gae + exp.Value;
                returns.Insert(0, returnValue);

                nextValue = exp.Value;
            }

            // Normalize advantages
            float meanAdvantage = advantages.Average();
            float stdAdvantage = (float)Math.Sqrt(advantages.Select(a => Math.Pow(a - meanAdvantage, 2)).Average() + 1e-8f);
            for (int i = 0; i < advantages.Count; i++)
                advantages[i] = (advantages[i] - meanAdvantage) / (stdAdvantage + (float)1e-8);
        }

        private void UpdateNetworks(List<Experience> experiences, List<float> returns, List<float> advantages, List<int> batchIndices)
        {
            // Initialize gradient accumulators
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

                // === Actor (policy) loss ===
                Vector<float> logits = actorNetwork.Forward(exp.State);
                Vector<float> probs = Softmax(logits);

                float newLogProb = (float)Math.Log(probs[exp.Action] + 1e-10f);
                float ratio = (float)Math.Exp(newLogProb - exp.LogProbability);
                float surrogate1 = ratio * advantage;
                float surrogate2 = Math.Clamp(ratio, 1 - clipEpsilon, 1 + clipEpsilon) * advantage;
                float policyLoss = -Math.Min(surrogate1, surrogate2);

                // Entropy bonus
                float entropy = -probs.Sum(p => p * (float)Math.Log(p + 1e-10f));

                // === Critic (value) loss ===
                float value = criticNetwork.Forward(exp.State)[0];
                float v_pred_clipped = exp.Value + Math.Clamp(value - exp.Value, -clipEpsilon, clipEpsilon);
                float v_loss1 = MathF.Pow(value - target, 2);
                float v_loss2 = MathF.Pow(v_pred_clipped - target, 2);
                float valueLoss = valueCoeff * Math.Max(v_loss1, v_loss2);

                // === Accumulate total losses (for logging only) ===
                actorLoss += policyLoss - entropyCoeff * entropy;
                criticLoss += valueLoss;

                // === Calculate gradients and backprop ===
                CalculateGradients(
                    exp,
                    advantage,
                    target,
                    actorGradWeights,
                    actorGradBiases,
                    criticGradWeights,
                    criticGradBiases
                );
            }

            // === Normalize gradients by batch size ===
            NormalizeGradients(actorGradWeights, actorGradBiases, batchIndices.Count);
            NormalizeGradients(criticGradWeights, criticGradBiases, batchIndices.Count);

            // === Optional: Clip gradients by global norm ===
            float maxGradNorm = 0.5f;  // or tune this value
            ClipGradients(actorGradWeights, actorGradBiases, maxGradNorm);
            ClipGradients(criticGradWeights, criticGradBiases, maxGradNorm);

            // === Apply optimizer updates ===
            actorOptimizer.Update(actorNetwork.GetWeights(), actorNetwork.GetBiases(), actorGradWeights, actorGradBiases);
            criticOptimizer.Update(criticNetwork.GetWeights(), criticNetwork.GetBiases(), criticGradWeights, criticGradBiases);

            // === Logging (optional) ===
            Console.WriteLine($"[PPO] Actor Loss: {actorLoss / batchIndices.Count}, Critic Loss: {criticLoss / batchIndices.Count}");


        }

        private List<Matrix<float>> InitializeGradients(List<Matrix<float>> weights)
        {
            return weights.Select(w => Matrix<float>.Build.Dense(w.RowCount, w.ColumnCount, 0f)).ToList();
        }

        private List<Vector<float>> InitializeGradients(List<Vector<float>> biases)
        {
            return biases.Select(b => Vector<float>.Build.Dense(b.Count, 0f)).ToList();
        }

        private void NormalizeGradients(List<Matrix<float>> gradWeights, List<Vector<float>> gradBiases, float batchSize)
        {
            for (int i = 0; i < gradWeights.Count; i++)
            {
                gradWeights[i] /= batchSize;
                gradBiases[i] /= batchSize;
            }
        }

        private void ClipGradients(List<Matrix<float>> gradWeights, List<Vector<float>> gradBiases, float maxNorm)
        {
            float totalNorm = 0f;

            foreach (var gw in gradWeights)
                totalNorm += gw.PointwisePower(2).Enumerate().Sum();

            foreach (var gb in gradBiases)
                totalNorm += gb.PointwisePower(2).Sum();

            totalNorm = MathF.Sqrt(totalNorm);

            if (totalNorm > maxNorm)
            {
                float scale = maxNorm / (totalNorm + 1e-6f);  // avoid div zero

                for (int i = 0; i < gradWeights.Count; i++)
                    gradWeights[i].MapInplace(x => x * scale);

                for (int i = 0; i < gradBiases.Count; i++)
                    gradBiases[i].MapInplace(x => x * scale);
            }
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
            // === ACTOR GRADIENTS ===

            // Forward pass to get logits
            Vector<float> actorOutput = actorNetwork.Forward(exp.State);

            // Convert logits to probabilities (softmax)
            Vector<float> probs = Softmax(actorOutput);

            // Calculate gradient of log π(a|s)
            Vector<float> gradLogPi = Vector<float>.Build.Dense(probs.Count, i => (i == exp.Action ? 1f : 0f) - probs[i]);

            // Weight by advantage (PPO surrogate signal)
            Vector<float> actorGrad = gradLogPi * advantage;

            // Add entropy gradient to encourage exploration
            Vector<float> entropyGrad = probs.Map(p => -MathF.Log(p + 1e-10f) - 1f) * entropyCoeff;

            // Combine policy + entropy gradients
            actorGrad += entropyGrad;

            // Backpropagate actor gradients
            BackpropagateGradients(actorNetwork, exp.State, actorGrad, actorGradWeights, actorGradBiases);

            // === CRITIC GRADIENTS ===

            // Forward pass to get value estimate
            Vector<float> criticOutput = criticNetwork.Forward(exp.State);
            float valueEstimate = criticOutput[0];

            // Compute critic gradient: ∇ (V(s) - target)^2 = 2 * (V(s) - target)
            float valueError = valueEstimate - target;
            Vector<float> criticGrad = Vector<float>.Build.Dense(1, 2f * valueError * valueCoeff);

            // Backpropagate critic gradients
            BackpropagateGradients(criticNetwork, exp.State, criticGrad, criticGradWeights, criticGradBiases);
        }

        private void BackpropagateGradients(
    NeuralNetwork network,
    Vector<float> input,
    Vector<float> outputGrad,  // ∂L/∂output, shape = last layer size
    List<Matrix<float>> gradWeights,
    List<Vector<float>> gradBiases)
        {
            var weights = network.GetWeights();
            var biases = network.GetBiases();

            var activations = new List<Vector<float>> { input };
            var zs = new List<Vector<float>>();

            Vector<float> activation = input;

            // === Forward pass: collect activations and pre-activations (zs) ===
            for (int i = 0; i < weights.Count; i++)
            {
                var z = weights[i].Multiply(activation).Add(biases[i]);
                zs.Add(z);

                // Apply hidden activations (ReLU)
                if (i < weights.Count - 1)
                {
                    activation = ApplyReLU(z);
                }
                else
                {
                    // Output layer: check for non-linear activations if used
                    // Here we assume linear, but you can add tanh/sigmoid handling if needed
                    activation = z;
                }

                activations.Add(activation);
            }

            // === Backward pass ===
            Vector<float> delta = outputGrad;

            for (int layer = weights.Count - 1; layer >= 0; layer--)
            {
                var aPrev = activations[layer];      // input to this layer
                var zCurrent = zs[layer];            // pre-activation of this layer

                if (layer < weights.Count - 1)
                {
                    // Hidden layers: apply ReLU derivative
                    delta = delta.PointwiseMultiply(ApplyReLUDerivative(zCurrent));
                }
                else
                {
                    delta = delta.PointwiseMultiply(ApplyReLUDerivative(zCurrent));
                }

                // Accumulate gradients
                gradWeights[layer] += delta.OuterProduct(aPrev);
                gradBiases[layer] += delta;

                if (layer > 0)
                {
                    // Backpropagate to previous layer
                    delta = weights[layer].TransposeThisAndMultiply(delta);
                }
            }
        }

        private Vector<float> ApplyReLU(Vector<float> input)
        {
            return input.Map(x => Math.Max(0f, x));
        }

        private Vector<float> ApplyReLUDerivative(Vector<float> input)
        {
            return input.Map(x => x > 0f ? 1f : 0f);
        }

        private Vector<float> ApplyTanhDerivative(Vector<float> input)
        {
            return input.Map(x =>
            {
                float t = MathF.Tanh(x);
                return 1f - t * t;
            });
        }

        private Vector<float> ApplySigmoidDerivative(Vector<float> input)
        {
            return input.Map(x =>
            {
                float s = 1f / (1f + MathF.Exp(-x));
                return s * (1f - s);
            });
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
                string jsonFilePath = "ppo_model_episode.json"; // Replace with your actual JSON file path
                if (File.Exists(jsonFilePath))
                {
                    string jsonContent = File.ReadAllText(jsonFilePath);
                    var model = JsonSerializer.Deserialize<Dictionary<string, object>>(jsonContent);

                    if (totalReward.Last() > (model != null && model.TryGetValue("TOTAL_Reward", out var value) && value is JsonElement jsonElement && jsonElement.ValueKind == JsonValueKind.Array
                        ? jsonElement.EnumerateArray().Select(x => x.GetSingle()).Max()
                        : float.MinValue))
                    {
                        SaveModel(agent, $"ppo_model_episode.json", episode + 1);
                        Console.WriteLine($"Model saved at episode {episode + 1}");
                    }
                }
                else
                {
                    Console.WriteLine("JSON file not found.");
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
            int stateSize = 7;    // Adjust to match your environment
            int actionSize = 5;
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