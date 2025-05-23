#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using MathNet.Numerics.LinearAlgebra;
using Microsoft.Xna.Framework;


class PPOHelper
{
    private static Random random = new Random();

    /// <summary>
    /// Initializes a weight matrix with scaled random values.
    /// </summary>
    /// <param name="inputSize">Size of the input layer</param>
    /// <param name="outputSize">Size of the output layer</param>
    /// <returns>A 2D array of initialized weights scaled by sqrt(2/inputSize)</returns>
    public double[,] InitializeWeights(int inputSize, int outputSize)
    {
        double[,] weights = new double[inputSize, outputSize];
        double scale = Math.Sqrt(2.0 / inputSize); // He Initialization

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
    public double[] LinearLayer(double[] input, double[,] weights, double[]? biasWeights = null)
    {
        int inputSize = weights.GetLength(0);
        int outputSize = weights.GetLength(1);

        if (input.Length != inputSize)
            throw new ArgumentException("Input length does not match weight dimensions.");

        double[] output = new double[outputSize];

        for (int i = 0; i < outputSize; i++)
        {
            for (int j = 0; j < inputSize; j++)
            {
                output[i] += input[j] * weights[j, i];
            }

            if (biasWeights != null)
                output[i] += biasWeights[i];
        }

        return output;
    }

    /// <summary>
    /// Applies the LeakyReLU activation function element-wise to the input vector.
    /// LeakyReLU(x) = max(0, x)
    /// </summary>
    /// <param name="x">Input vector</param>
    /// <returns>Vector with LeakyReLU activation applied</returns>
    public double[] LeakyReLU(double[] input, double alpha = 0.01)
    {
        return input.Select(x => x > 0 ? x : alpha * x).ToArray();
    }
    public double[] ELU(double[] input, double alpha = 1.0)
    {
        return input.Select(x => (x > 0) ? x : alpha * (Math.Exp(x) - 1)).ToArray();
    }
    /// <summary>
    /// Applies the softmax function to convert logits to probabilities.
    /// </summary>
    /// <param name="x">Input logits</param>
    /// <returns>Probability distribution that sums to 1</returns>
    public double[] Softmax(double[] logits)
    {
        double maxLogit = logits.Max(); // Prevents numerical instability
        double[] expValues = logits.Select(l => Math.Exp((l - maxLogit))).ToArray();
        double sumExp = expValues.Sum();
        return expValues.Select(e => e / sumExp).ToArray();
    }

    public static double RandomGaussian(double mean = 0.0, double stdDev = 1.0)
    {
        double u1 = 1.0 - random.NextDouble(); // Uniform(0,1] random value
        double u2 = 1.0 - random.NextDouble();

        double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
        return mean + stdDev * randStdNormal; // Scale to desired mean/stdDev
    }

    // Helper method to get property with case-insensitive matching
    public JsonElement GetPropertyNew(JsonElement element, string propertyName)
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

    public double[][] ConvertToJaggedArray(double[,] array)
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

    public double[,] ConvertTo2DArray(double[][] jaggedArray)
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

    public double[][] SafeDeserialize2D(JsonElement root, string property)
    {
        return root.TryGetProperty(property, out JsonElement element) && element.ValueKind != JsonValueKind.Null
            ? JsonSerializer.Deserialize<double[][]>(element.GetRawText()) ?? new double[0][]
            : new double[0][];
    }

    public double[] SafeDeserialize1D(JsonElement root, string property)
    {
        return root.TryGetProperty(property, out JsonElement element) && element.ValueKind != JsonValueKind.Null
            ? JsonSerializer.Deserialize<double[]>(element.GetRawText()) ?? new double[0]
            : new double[0];
    }

    public List<Matrix<float>> ConvertToMatrices(List<List<List<float>>> jagged)
    {
        return jagged.Select(layer =>
            Matrix<float>.Build.DenseOfRows(layer.Select(row =>
                Vector<float>.Build.DenseOfArray(row.ToArray()))
            )
        ).ToList();
    }

    public List<Vector<float>> ConvertToVectors(List<List<float>> jagged)
    {
        return jagged.Select(layer =>
            Vector<float>.Build.DenseOfArray(layer.ToArray())
        ).ToList();
    }


    /// <summary>
    /// Samples an action from a probability distribution over actions.
    /// </summary>
    /// <param name="actionProbs">Probability distribution over actions</param>
    /// <returns>Chosen action index</returns>
    public int SampleAction(double[] actionProbs, bool isTraining = true)
    {
        double temperature = isTraining ? 1.0 : 0.5;

        if (temperature != 1.0)
        {
            double[] scaledProbs = actionProbs.Select(p => Math.Pow(p, 1 / temperature)).ToArray();
            double sum = scaledProbs.Sum();
            actionProbs = scaledProbs.Select(p => p / sum).ToArray();
        }

        // Ensure normalization
        double total = actionProbs.Sum();
        if (Math.Abs(total - 1.0) > 1e-6)
        {
            actionProbs = actionProbs.Select(p => p / total).ToArray();
        }

        double sample = random.NextDouble();
        double cumulative = 0;

        for (int i = 0; i < actionProbs.Length; i++)
        {
            cumulative += actionProbs[i];
            if (sample <= cumulative)
                return i;
        }

        return actionProbs.Select((p, i) => (p, i)).OrderByDescending(x => x.p).First().i;
    }

    public double CalculateLayerGradient(double weight)
    {
        // Add small noise for exploration
        double noise = random.NextDouble() * 0.01;

        // Calculate gradient with weight decay
        double weightDecay = 0.0001 * weight;

        return weight + noise + weightDecay;
    }


    public double CalculateOutputGradient(double weight)
    {
        // Similar to layer gradient but with different scaling
        double noise = random.NextDouble() * 0.005;
        double weightDecay = 0.0001 * weight;

        return weight + noise + weightDecay;
    }

    public double CalculateEntropyBonus(double[] actionProbabilities)
    {
        double entropy = 0.0;

        // Calculate entropy: -Σ P(a) * log(P(a))
        foreach (double prob in actionProbabilities)
        {
            if (prob > 1e-10)  // Avoid log(0), use small epsilon
            {
                entropy -= prob * Math.Log(prob);
            }
        }

        return entropy;
    }

    // Computes the PPO gradient with entropy regularization
    public double ComputePPOGradient(double weight, double loss, double entropyBonus)
    {
        double entropyTerm = entropyBonus > 0 ? entropyBonus : 0; // Entropy regularization
        return loss * (weight + entropyTerm);
    }

    // Applies gradient norm clipping to prevent exploding gradients
    public double ClipGradient(double gradient)
    {
        double maxNorm = 1.0;  // Set the max allowed gradient norm
        if (Math.Abs(gradient) > maxNorm)
        {
            return Math.Sign(gradient) * maxNorm;
        }
        return gradient;
    }

    public double[,] InitializeVelocity(double[,] weights)
    {
        Random rand = new Random();
        double[,] velocity = new double[weights.GetLength(0), weights.GetLength(1)];

        for (int i = 0; i < weights.GetLength(0); i++)
        {
            for (int j = 0; j < weights.GetLength(1); j++)
            {
                velocity[i, j] = (rand.NextDouble() * 0.2) - 0.1; // Range: [-0.1, 0.1]
            }
        }
        return velocity;
    }

    public List<float[][]> ConvertToJaggedList(List<Matrix<float>> matrices)
    {
        var jaggedList = new List<float[][]>();

        foreach (var matrix in matrices)
        {
            var rows = matrix.RowCount;
            var cols = matrix.ColumnCount;
            var jagged = new float[rows][];

            for (int i = 0; i < rows; i++)
            {
                jagged[i] = new float[cols];
                for (int j = 0; j < cols; j++)
                {
                    jagged[i][j] = matrix[i, j];
                }
            }

            jaggedList.Add(jagged);
        }

        return jaggedList;
    }
    public List<float[]> ConvertVectorsToJaggedList(List<Vector<float>> vectors)
    {
        var result = new List<float[]>();

        foreach (var vector in vectors)
        {
            result.Add(vector.ToArray());
        }

        return result;
    }
    public double[] InitializeVelocity(double[] biases)
    {
        Random rand = new Random();
        double[] velocity = new double[biases.Length];

        for (int i = 0; i < biases.Length; i++)
        {
            velocity[i] = (rand.NextDouble() * 0.2) - 0.1; // Range: [-0.1, 0.1]
        }
        return velocity;
    }
}
