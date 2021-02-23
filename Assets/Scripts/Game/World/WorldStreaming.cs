using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldStreaming
{
    private static readonly float[] CHUNK_LOD_DIST =
    {
        WorldChunk.CHUNK_SIZE * 2,
        WorldChunk.CHUNK_SIZE * 4,
        WorldChunk.CHUNK_SIZE * 6
    };

    private const int LOD_UPDATE_SLICES = 5;

    private class LoadedChunkMeshParams
    {
        public NativeArray<Vector3> verts;
        public NativeArray<int> tris;
        public NativeArray<Vector2> uvs;
        public NativeArray<Vector3> normals;
    }

    private class LoadedChunk
    {
        public int id;
        public bool unload = false;

        public GameObject obj;
        public GameObject[] lods;
        public int lod_frame_idx;
        public LoadedChunkMeshParams[] mesh_params;

        public bool load_started = false;
        public bool load_finalized = false;
        public JobHandle load_job;
    }

    private Dictionary<int, LoadedChunk> m_loaded_chunks = new Dictionary<int, LoadedChunk>();

    private int m_chunk_id_last = -1;

    private World m_world;
    private WorldGen m_world_gen;

    private static ProfilerMarker s_zone_transition_marker = new ProfilerMarker("WorldStreaming.HandleZoneTransition");
    private static ProfilerMarker s_zone_finalization_marker = new ProfilerMarker("WorldStreaming.HandleZoneFinalization");
    private static ProfilerMarker s_lod_marker = new ProfilerMarker("WorldStreaming.SelectActiveLod");

    public WorldStreaming(World world, WorldGen world_gen)
    {
        m_world = world;
        m_world_gen = world_gen;
    }

    public void Update(Vector3 position)
    {
        using (s_zone_transition_marker.Auto())
        {
            int chunk_id = GetChunkId(position.x, position.z);
            if (chunk_id != m_chunk_id_last)
            {
                HandleZoneTransition(position);
                m_chunk_id_last = chunk_id;
            }
        }

        using (s_zone_finalization_marker.Auto())
        {
            const float FINALIZATION_QUANTUM = 2.0f / 1000.0f;
            HandleZoneFinalization(FINALIZATION_QUANTUM);
        }

        using (s_lod_marker.Auto())
        {
            foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
            {
                if (chunk.Value.load_finalized &&
                    Time.frameCount % LOD_UPDATE_SLICES == chunk.Value.lod_frame_idx)
                {
                    SelectActiveLod(position, chunk.Value);
                }
            }
        }
    }

    private const int CHUNKS_PER_ROW = (int)WorldGen.MAX_WORLD_COORDINATE * 2 / WorldChunk.CHUNK_SIZE;

    public static int GetChunkId(float x, float z)
    {
        x += WorldGen.MAX_WORLD_COORDINATE;
        z += WorldGen.MAX_WORLD_COORDINATE;

        int x_grid = (int)x / WorldChunk.CHUNK_SIZE;
        int z_grid = (int)z / WorldChunk.CHUNK_SIZE;

        return x_grid + z_grid * CHUNKS_PER_ROW;
    }

    public static Vector2 GetChunkCoords(int chunk_id)
    {
        float x = -WorldGen.MAX_WORLD_COORDINATE;
        float z = -WorldGen.MAX_WORLD_COORDINATE;

        x += (chunk_id % CHUNKS_PER_ROW) * WorldChunk.CHUNK_SIZE;
        z += (chunk_id / CHUNKS_PER_ROW) * WorldChunk.CHUNK_SIZE;

        return new Vector2(x, z);
    }

    private List<int> GetTargetChunks(Vector3 position)
    {
        float furthest_radius = CHUNK_LOD_DIST[CHUNK_LOD_DIST.Length - 1];

        Vector2 top_left = new Vector2(position.x - furthest_radius - WorldChunk.CHUNK_SIZE, position.z - furthest_radius - WorldChunk.CHUNK_SIZE);
        Vector2 bottom_right = new Vector2(position.x + furthest_radius + WorldChunk.CHUNK_SIZE, position.z + furthest_radius + WorldChunk.CHUNK_SIZE);

        // align to chunk boundaries
        top_left.x = WorldChunk.CHUNK_SIZE * (int)(top_left.x / WorldChunk.CHUNK_SIZE);
        top_left.y = WorldChunk.CHUNK_SIZE * (int)(top_left.y / WorldChunk.CHUNK_SIZE);

        bottom_right.x = WorldChunk.CHUNK_SIZE * (int)(bottom_right.x / WorldChunk.CHUNK_SIZE);
        bottom_right.y = WorldChunk.CHUNK_SIZE * (int)(bottom_right.y / WorldChunk.CHUNK_SIZE);

        Assert.IsTrue(top_left.x % WorldChunk.CHUNK_SIZE == 0.0f);
        Assert.IsTrue(top_left.y % WorldChunk.CHUNK_SIZE == 0.0f);
        Assert.IsTrue(bottom_right.x % WorldChunk.CHUNK_SIZE == 0.0f);
        Assert.IsTrue(bottom_right.x % WorldChunk.CHUNK_SIZE == 0.0f);

        List<int> chunks = new List<int>();

        for (float z = top_left.y; z < bottom_right.y; z += WorldChunk.CHUNK_SIZE)
        {
            for (float x = top_left.x; x < bottom_right.x; x += WorldChunk.CHUNK_SIZE)
            {
                chunks.Add(GetChunkId(x, z));
            }
        }

        return chunks;
    }

    private JobHandle ScheduleChunkLoad(int chunk_id, out LoadedChunkMeshParams[] mesh_params)
    {
        Vector2 position = GetChunkCoords(chunk_id);

        const int LOD_COUNT = (int)WorldChunk.Lod.EnumCount;
        mesh_params = new LoadedChunkMeshParams[LOD_COUNT];

        NativeArray<JobHandle> handles = new NativeArray<JobHandle>(LOD_COUNT, Allocator.TempJob);

        for (int i = 0; i < LOD_COUNT; ++i)
        {
            NativeArray<float> heights;
            JobHandle chunk_gen_handle = WorldChunk.ScheduleChunkGeneration((int)position.x, (int)position.y, (WorldChunk.Lod)i, m_world_gen.GetPerlin(), out heights);

            LoadedChunkMeshParams param = new LoadedChunkMeshParams();
            handles[i] = WorldChunk.ScheduleChunkMeshGeneration(heights, out param.verts, out param.tris, out param.uvs, out param.normals, chunk_gen_handle);
            mesh_params[i] = param;
        }

        JobHandle deps = JobHandle.CombineDependencies(handles);
        handles.Dispose();
        return deps;
    }

    private void FreeMeshParams(LoadedChunkMeshParams[] mesh_params)
    {
        foreach (LoadedChunkMeshParams param in mesh_params)
        {
            param.verts.Dispose();
            param.tris.Dispose();
            param.uvs.Dispose();
            param.normals.Dispose();
        }
    }

    private void SelectActiveLod(Vector3 pos, LoadedChunk chunk)
    {
        float dist_squared = (new Vector2(pos.x, pos.z) - GetChunkCoords(chunk.id)).sqrMagnitude;

        int selected_lod_idx = -1;

        for (int i = 0; i < CHUNK_LOD_DIST.Length; ++i)
        {
            float lod_switch_squared = CHUNK_LOD_DIST[i] * CHUNK_LOD_DIST[i];

            if (dist_squared < lod_switch_squared)
            {
                selected_lod_idx = i;
                break;
            }
        }

        chunk.obj.SetActive(selected_lod_idx != -1);

        for (int i = 0; i < chunk.lods.Length; ++i)
        {
            chunk.lods[i].SetActive(i == selected_lod_idx);
        }
    }

    private void HandleZoneTransition(Vector3 position)
    {
        List<int> chunks = GetTargetChunks(position);

        foreach (int chunk in chunks)
        {
            if (!m_loaded_chunks.ContainsKey(chunk))
            {
                m_loaded_chunks.Add(chunk, new LoadedChunk { id = chunk });
            }
        }

        foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
        {
            bool needs_unload = true;

            for (int i = 0; i < chunks.Count; ++i)
            {
                if (chunks[i] == chunk.Key)
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
    }

    private void HandleZoneFinalization(float timeslice)
    {
        float time_start = Time.realtimeSinceStartup;

        List<int> unloaded_chunks = new List<int>();

        foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
        {
            if (!chunk.Value.load_started)
            {
                chunk.Value.load_started = true;
                chunk.Value.load_job = ScheduleChunkLoad(chunk.Key, out chunk.Value.mesh_params);
            }
            else if (chunk.Value.unload)
            {
                if (chunk.Value.load_finalized) // finalized, we only need to free the objects
                {
                    Object.Destroy(chunk.Value.obj);
                    foreach (Object obj in chunk.Value.lods)
                    {
                        Object.Destroy(obj);
                    }
                    unloaded_chunks.Add(chunk.Key);
                }
                else if (chunk.Value.load_job.IsCompleted) // not finalized, but tasks done - we can release resources
                {
                    chunk.Value.load_job.Complete();
                    FreeMeshParams(chunk.Value.mesh_params);
                    unloaded_chunks.Add(chunk.Key);
                }
            }
            else if (!chunk.Value.load_finalized && chunk.Value.load_job.IsCompleted)
            {
                chunk.Value.load_job.Complete();
                chunk.Value.load_finalized = true;

                Vector2 chunk_pos = GetChunkCoords(chunk.Key);

                GameObject obj = new GameObject(string.Format("Chunk{0}", chunk.Key));
                obj.transform.parent = m_world.gameObject.transform;
                obj.transform.position = new Vector3(chunk_pos.x, 0.0f, chunk_pos.y);
                chunk.Value.obj = obj;

                const int LOD_COUNT = (int)WorldChunk.Lod.EnumCount;
                chunk.Value.lods = new GameObject[LOD_COUNT];

                for (int i = 0; i < LOD_COUNT; ++i)
                {
                    GameObject lod_obj = new GameObject(string.Format("Lod{1}", chunk.Key, i));
                    lod_obj.transform.parent = obj.transform;
                    lod_obj.transform.localPosition = new Vector3(0.0f, 0.0f, 0.0f);

                    MeshRenderer meshRenderer = lod_obj.AddComponent<MeshRenderer>();
                    meshRenderer.sharedMaterial = Resources.Load<Material>("GrassMaterial");
                    meshRenderer.sharedMaterial.SetTextureScale("_BaseMap", new Vector2(WorldChunk.CHUNK_SIZE / 64.0f, WorldChunk.CHUNK_SIZE / 64.0f));

                    Assert.IsTrue(chunk.Value.mesh_params.Length == LOD_COUNT);
                    LoadedChunkMeshParams param = chunk.Value.mesh_params[i];

                    MeshFilter meshFilter = lod_obj.AddComponent<MeshFilter>();
                    meshFilter.mesh = WorldChunk.FinalizeChunkMesh(param.verts, param.tris, param.uvs, param.normals);

                    chunk.Value.lods[i] = lod_obj;
                }

                chunk.Value.lod_frame_idx = Time.frameCount % LOD_UPDATE_SLICES;

                FreeMeshParams(chunk.Value.mesh_params);
            }

            if (Time.realtimeSinceStartup - time_start > timeslice)
            {
                break;
            }
        }

        foreach (int chunk in unloaded_chunks)
        {
            m_loaded_chunks.Remove(chunk);
        }
    }
}
