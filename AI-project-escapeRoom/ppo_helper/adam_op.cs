using System;
using System.Collections.Generic;

public class AdamOptimizer
{
    private double learningRate;
    private double beta1;
    private double beta2;
    private double epsilon;
    private int timestep;

    private Dictionary<string, double[]> m;
    private Dictionary<string, double[]> v;

    public AdamOptimizer(double learningRate = 0.001, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
    {
        this.learningRate = learningRate;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.timestep = 0;

        m = new Dictionary<string, double[]>();
        v = new Dictionary<string, double[]>();
    }

    public void Update(string paramName, double[] param, double[] grad)
    {
        if (!m.ContainsKey(paramName))
        {
            m[paramName] = new double[param.Length];
            v[paramName] = new double[param.Length];
        }

        timestep++;

        double[] mParam = m[paramName];
        double[] vParam = v[paramName];

        for (int i = 0; i < param.Length; i++)
        {
            // Update biased first moment estimate
            mParam[i] = beta1 * mParam[i] + (1 - beta1) * grad[i];

            // Update biased second raw moment estimate
            vParam[i] = beta2 * vParam[i] + (1 - beta2) * grad[i] * grad[i];

            // Compute bias-corrected first and second moment estimates
            double mHat = mParam[i] / (1 - Math.Pow(beta1, timestep));
            double vHat = vParam[i] / (1 - Math.Pow(beta2, timestep));

            // Update parameter
            param[i] -= learningRate * mHat / (Math.Sqrt(vHat) + epsilon);
        }
    }
}
