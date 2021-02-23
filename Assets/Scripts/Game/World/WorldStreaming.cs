using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

public class WorldStreaming
{
    private class TargetChunk
    {
        public int id;
    }

    private class LoadedChunk
    {
        public TargetChunk chunk;
        public GameObject obj;
        public bool unload = false;

        public bool load_finalized = false;
        public JobHandle load_job;
        public NativeArray<Vector3> load_verts;
        public NativeArray<int> load_tris;
        public NativeArray<Vector2> load_uvs;
        public NativeArray<Vector3> load_normals;
    }

    private Dictionary<int, LoadedChunk> m_loaded_chunks = new Dictionary<int, LoadedChunk>();

    private int m_chunk_id_last = -1;

    private const float QUANTUM = 2.0f / 1000.0f;

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

            // -- Load any chunks that need loading

            foreach (TargetChunk chunk in chunks)
            {
                LoadedChunk loaded_chunk;

                bool need_load = true;

                if (m_loaded_chunks.TryGetValue(chunk.id, out loaded_chunk))
                {
                    need_load = false;
                }
                else
                {
                    loaded_chunk = new LoadedChunk();
                    loaded_chunk.chunk = chunk;
                    m_loaded_chunks.Add(chunk.id, loaded_chunk);
                }

                if (need_load)
                {
                    loaded_chunk.load_job = ScheduleChunkLoad(
                        chunk.id,
                        WorldChunk.Lod.FullQuality,
                        out loaded_chunk.load_verts,
                        out loaded_chunk.load_tris,
                        out loaded_chunk.load_uvs,
                        out loaded_chunk.load_normals);
                }
            }

            // -- Find any chunks that we want to unload, and unload them.

            foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
            {
                bool needs_unload = true;

                for (int i = 0; i < chunks.Count; ++i)
                {
                    if (chunks[i].id == chunk.Key)
                    {
                        needs_unload = false;
                        break;
                    }
                }

                if (needs_unload)
                {
                    chunk.Value.unload = true;
                }
            }

            m_chunk_id_last = chunk_id;
        }

        float time_start = Time.realtimeSinceStartup;

        List<int> unloaded_chunks = new List<int>();

        foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
        {
            if (chunk.Value.unload)
            {
                if (chunk.Value.load_finalized) // finalized, we only need to free the object
                {
                    Object.Destroy(chunk.Value.obj);
                    unloaded_chunks.Add(chunk.Key);
                }
                else if (chunk.Value.load_job.IsCompleted) // not finalized, but tasks done - we can release resources
                {
                    chunk.Value.load_job.Complete();
                    chunk.Value.load_verts.Dispose();
                    chunk.Value.load_tris.Dispose();
                    chunk.Value.load_uvs.Dispose();
                    chunk.Value.load_normals.Dispose();
                    unloaded_chunks.Add(chunk.Key);
                }
            }
            else if (!chunk.Value.load_finalized && chunk.Value.load_job.IsCompleted)
            {
                chunk.Value.load_job.Complete();
                chunk.Value.load_finalized = true;

                GameObject obj = new GameObject(string.Format("WorldChunk_{0}", chunk.Key));

                Vector2 chunk_pos = GetChunkCoords(chunk.Value.chunk.id);
                obj.transform.position = new Vector3(chunk_pos.x, 0.0f, chunk_pos.y);

                MeshRenderer meshRenderer = obj.AddComponent<MeshRenderer>();
                meshRenderer.sharedMaterial = Resources.Load<Material>("GrassMaterial");
                meshRenderer.sharedMaterial.SetTextureScale("_BaseMap", new Vector2(WorldChunk.CHUNK_SIZE / 64.0f, WorldChunk.CHUNK_SIZE / 64.0f));

                MeshFilter meshFilter = obj.AddComponent<MeshFilter>();
                meshFilter.mesh = WorldChunk.FinalizeChunkMesh(
                    chunk.Value.load_verts,
                    chunk.Value.load_tris,
                    chunk.Value.load_uvs,
                    chunk.Value.load_normals);

                chunk.Value.load_verts.Dispose();
                chunk.Value.load_tris.Dispose();
                chunk.Value.load_uvs.Dispose();
                chunk.Value.load_normals.Dispose();

                chunk.Value.obj = obj;
            }

            float time_now = Time.realtimeSinceStartup;
            float time_delta = time_now - time_start;
            if (time_delta > QUANTUM)
            {
                break;
            }
        }

        foreach (int chunk in unloaded_chunks)
        {
            m_loaded_chunks.Remove(chunk);
        }
    }

    private const int CHUNKS_PER_ROW = (int)WorldGen.MAX_WORLD_COORDINATE * 2 / WorldChunk.CHUNK_SIZE;

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
                TargetChunk target = new TargetChunk();
                target.id = GetChunkId(x, z);
                chunks.Add(target);
            }
        }

        return chunks;
    }

    private JobHandle ScheduleChunkLoad(int chunk_id, WorldChunk.Lod quality, out NativeArray<Vector3> verts, out NativeArray<int> tris, out NativeArray<Vector2> uvs, out NativeArray<Vector3> normals)
    {
        Vector2 position = GetChunkCoords(chunk_id);
        NativeArray<float> heights;
        JobHandle chunk_gen_handle = WorldChunk.ScheduleChunkGeneration((int)position.x, (int)position.y, quality, m_world_gen.GetPerlin(), out heights);
        return WorldChunk.ScheduleChunkMeshGeneration(heights, out verts, out tris, out uvs, out normals, chunk_gen_handle);
    }
}
