using System.Collections.Generic;

/// <summary>
/// Loads training progress from a file.
/// </summary>
/// <param name="filePath">Path to the progress file</param>
/// <returns>Episode number, best reward, and recent rewards</returns>
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