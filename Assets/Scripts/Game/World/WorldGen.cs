using UnityEngine;
using UnityEngine.Assertions;

public struct PerlinGenerator
{
    private float m_x_offset;
    private float m_z_offset;

    public PerlinGenerator(float x_offset, float z_offset)
    {
        m_x_offset = x_offset;
        m_z_offset = z_offset;
    }

    public float GetHeightAt(float x, float z)
    {
        Assert.IsTrue(x >= -WorldGen.MAX_WORLD_COORDINATE && x <= WorldGen.MAX_WORLD_COORDINATE);
        Assert.IsTrue(z >= -WorldGen.MAX_WORLD_COORDINATE && z <= WorldGen.MAX_WORLD_COORDINATE);
        float height = Mathf.PerlinNoise(x + m_x_offset, z + m_z_offset);
        return height;
    }
}

public class WorldGen
{
    public const int VERSION = 1;

    public const float MAX_WORLD_COORDINATE = 32768;

    private int m_seed;

    private PerlinGenerator m_generator;

    public WorldGen(string seed)
    {
        m_seed = seed.GetStableHashCode();

        Random.State old_state = Random.state;
        Random.InitState(m_seed);
        m_generator = new PerlinGenerator(Random.Range(0.0f, 100.0f), Random.Range(0.0f, 100.0f));
        Random.state = old_state;
    }

    public PerlinGenerator GetPerlin()
    {
        return m_generator;
    }
}
