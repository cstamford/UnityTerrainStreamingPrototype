using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
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
    
    [BurstCompile]
    private struct GenerateChunkJob : IJobParallelFor
    {
        [DeallocateOnJobCompletion]
        [ReadOnly] public NativeArray<Vector2> coords;
        [ReadOnly] public PerlinGenerator perlin;

        public NativeArray<float> heights;

        public void Execute(int id)
        {
            heights[id] =
                perlin.GetHeightAt(coords[id].x / 64.0f, coords[id].y / 128.0f) * 32.0f +
                Mathf.Min(
                    perlin.GetHeightAt((coords[id].x + 0.05f) / 16.0f, (coords[id].y + 0.05f) / 32.0f),
                    perlin.GetHeightAt((coords[id].x - 0.05f) / 32.0f, (coords[id].y - 0.05f) / 16.0f));
        }
    }

    public JobHandle ScheduleBaseTerrainGeneration(int chunk_x, int chunk_z, int cell_size, out NativeArray<float> heights)
    {
        int heights_count = cell_size * cell_size;
        NativeArray<Vector2> coords = new NativeArray<Vector2>(heights_count, Allocator.TempJob);

        int i = 0;
        for (int z = chunk_z; z < chunk_z + cell_size; ++z)
        {
            for (int x = chunk_x; x < chunk_x + cell_size; ++x)
            {
                coords[i++] = new Vector2(x, z);
            }
        }

        heights = new NativeArray<float>(coords.Length, Allocator.Persistent);

        GenerateChunkJob job = new GenerateChunkJob()
        {
            coords = coords,
            perlin = m_generator,
            heights = heights
        };

        return job.Schedule(coords.Length, 32);
    }
}
