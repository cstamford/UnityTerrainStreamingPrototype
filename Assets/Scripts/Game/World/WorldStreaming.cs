using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldStreaming
{
    private const int CHUNKS_PER_ROW = (int)WorldGen.MAX_WORLD_COORDINATE * 2 / WorldChunk.CHUNK_SIZE;

    private class TargetChunk
    {
        public int id;
        public WorldChunk.Lod quality;
    }

    private class LoadedChunk
    {
        public TargetChunk chunk;
        public GameObject obj;

        public bool load_complete;
        public JobHandle load_job;
        public NativeArray<float> load_heights;
    }

    private Dictionary<int, LoadedChunk> m_loaded_chunks = new Dictionary<int, LoadedChunk>();

    private int m_chunk_id_last = -1;

    private WorldGen m_world_gen;

    public WorldStreaming(WorldGen world_gen)
    {
        m_world_gen = world_gen;
    }

    public void Update(Vector3 position)
    {
        int chunk_id = GetChunkId(position.x, position.z);
        if (chunk_id != m_chunk_id_last)
        {
            List<TargetChunk> chunks = GetTargetChunks(chunk_id);

            foreach (TargetChunk chunk in chunks)
            {
                LoadedChunk loaded_chunk;

                bool need_load = true;

                if (m_loaded_chunks.TryGetValue(chunk.id, out loaded_chunk))
                {
                    Assert.IsTrue(loaded_chunk.load_complete);

                    if (loaded_chunk.chunk.quality == chunk.quality)
                    {
                        need_load = false;
                    }
                }
                else
                {
                    loaded_chunk = new LoadedChunk();
                    loaded_chunk.chunk = chunk;
                    m_loaded_chunks.Add(chunk.id, loaded_chunk);
                }

                if (need_load)
                {
                    loaded_chunk.load_complete = false;
                    loaded_chunk.load_job = ScheduleChunkLoad(chunk.id, chunk.quality, out loaded_chunk.load_heights);
                }
            }

            // TODO: Unload any chunks that are not in the target chunk collection

            m_chunk_id_last = chunk_id;
        }

        foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
        {
            if (!chunk.Value.load_complete && chunk.Value.load_job.IsCompleted)
            {
                chunk.Value.load_job.Complete();
                chunk.Value.load_complete = true;

                GameObject chunk_obj = new GameObject(string.Format("WorldChunk_{0}", chunk.Key));

                Vector2 chunk_pos = GetChunkCoords(chunk.Value.chunk.id);
                chunk_obj.transform.position = new Vector3(chunk_pos.x, 0.0f, chunk_pos.y);

                MeshRenderer meshRenderer = chunk_obj.AddComponent<MeshRenderer>();
                meshRenderer.sharedMaterial = new Material(Shader.Find("Universal Render Pipeline/Simple Lit"));
                //meshRenderer.sharedMaterial.SetTexture("_BaseMap", Resources.Load<Texture>("GrassTexture"));
                //meshRenderer.sharedMaterial.SetTextureScale("_BaseMap", new Vector2(WorldChunk.CHUNK_SIZE / 4.0f, WorldChunk.CHUNK_SIZE / 4.0f));

                MeshFilter meshFilter = chunk_obj.AddComponent<MeshFilter>();
                meshFilter.mesh = WorldChunk.GenerateChunkMesh(chunk.Value.load_heights);

                chunk.Value.load_heights.Dispose();
                chunk.Value.obj = chunk_obj;
            }
        }
    }

    public static int GetChunkId(float x, float z)
    {
        x += WorldGen.MAX_WORLD_COORDINATE;
        z += WorldGen.MAX_WORLD_COORDINATE;

        int x_grid = (int)x / WorldChunk.CHUNK_SIZE;
        int z_grid = (int)z / WorldChunk.CHUNK_SIZE;

        return (int)(x_grid + z_grid * CHUNKS_PER_ROW);
    }

    public static Vector2 GetChunkCoords(int chunk_id)
    {
        float x = -WorldGen.MAX_WORLD_COORDINATE;
        float z = -WorldGen.MAX_WORLD_COORDINATE;

        x += (chunk_id % CHUNKS_PER_ROW) * WorldChunk.CHUNK_SIZE;
        z += (chunk_id / CHUNKS_PER_ROW) * WorldChunk.CHUNK_SIZE;

        return new Vector2(x, z);
    }

    private List<TargetChunk> GetTargetChunks(int chunk_id)
    {
        const int HIGH_QUALITY_CHUNKS = 2;
        const int MID_QUALITY_CHUNKS = 2;
        const int LOW_QUALITY_CHUNKS = 2;
        const int CHUNK_RADIUS = HIGH_QUALITY_CHUNKS + MID_QUALITY_CHUNKS + LOW_QUALITY_CHUNKS;

        List<TargetChunk> chunks = new List<TargetChunk>();

        Vector2 position = GetChunkCoords(chunk_id);

        Vector2 top_left = new Vector2(
            position.x - CHUNK_RADIUS * WorldChunk.CHUNK_SIZE, 
            position.y - CHUNK_RADIUS * WorldChunk.CHUNK_SIZE);

        Vector2 bottom_right = new Vector2(
            position.x + CHUNK_RADIUS * WorldChunk.CHUNK_SIZE,
            position.y + CHUNK_RADIUS * WorldChunk.CHUNK_SIZE);

        for (float z = top_left.y; z < bottom_right.y; z += WorldChunk.CHUNK_SIZE)
        {
            for (float x = top_left.x; x < bottom_right.x; x += WorldChunk.CHUNK_SIZE)
            {
                int distance_x = (int)(Mathf.Abs(position.x - x) / WorldChunk.CHUNK_SIZE);
                int distance_z = (int)(Mathf.Abs(position.y - z) / WorldChunk.CHUNK_SIZE);

                TargetChunk target = new TargetChunk();
                target.id = GetChunkId(x, z);

                int hq_cutoff = HIGH_QUALITY_CHUNKS;
                int mq_cutoff = HIGH_QUALITY_CHUNKS + MID_QUALITY_CHUNKS;

                if (distance_x <= hq_cutoff && distance_z <= hq_cutoff)
                {
                    target.quality = WorldChunk.Lod.FullQuality;
                }
                else if (distance_x <= mq_cutoff && distance_z <= mq_cutoff)
                {
                    target.quality = WorldChunk.Lod.FullQuality;//WorldChunk.Lod.HalfQuality;
                }
                else
                {
                    target.quality = WorldChunk.Lod.FullQuality;//WorldChunk.Lod.QuarterQuality;
                }

                chunks.Add(target);
            }
        }

        return chunks;
    }

    private JobHandle ScheduleChunkLoad(int chunk_id, WorldChunk.Lod quality, out NativeArray<float> heights)
    {
        Vector2 position = GetChunkCoords(chunk_id);
        return WorldChunk.ScheduleChunkGeneration((int)position.x, (int)position.y, quality, m_world_gen.GetPerlin(), out heights);
    }

}
