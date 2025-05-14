

using System.Collections.Generic;

public class ModelData
{
    public int IN_EPISODE { get; set; }
    public List<float> TOTAL_Reward { get; set; }
    public List<List<List<float>>> IN_ACTOR_WEIGHTS { get; set; }
    public List<List<float>> IN_ACTOR_BIASES { get; set; }
    public List<List<List<float>>> IN_CRITIC_WEIGHTS { get; set; }
    public List<List<float>> IN_CRITIC_BIASES { get; set; }
}