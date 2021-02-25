using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldStreaming : MonoBehaviour
{
    private const int LOD_COUNT = 4;

    private static readonly float[] LOD_SWITCH_DISTANCE =
    {
        128.0f,
        256.0f,
        512.0f,
        1024.0f
    };

    private static readonly int[] LOD_DISTANCE_PER_VERT =
    {
        1,
        2,
        16,
        WorldChunk.SIZE / 2
    };

    private static readonly Color[] LOD_DEBUG_COLOR =
    {
        Color.red,
        new Color(1.0f, 0.7f, 0.0f),
        Color.yellow,
        Color.green
    };

    private const int LOD_UPDATE_SLICES = LOD_COUNT;

    private class LoadedChunkMeshParams
    {
        public NativeArray<Vector3> verts;
        public NativeArray<int> tris;
        public NativeArray<Vector2> uvs;
        public NativeArray<Vector3> normals;
        public int final_vert_count;
        public int final_tri_count;
    }

    private class LodInfo
    {
        public GameObject obj;
        public GameObject skirts;
        public Vector3[] border_normals;
    }

    private class LoadedChunk
    {
        public bool unload = false;

        public GameObject obj;

        public LodInfo[] lod_info;
        public int lod_frame_idx = -1;
        public int lod_active_idx = -2;
        public LoadedChunkMeshParams[] lod_mesh_params;

        public bool load_started = false;
        public bool load_finalized = false;
        public JobHandle load_job;

        public NativeArray<float>[] nearby_heights;
    }

    public class LoadedChunkHeightData
    {
        public NativeArray<float> heights;
        public JobHandle job;
        public int ref_count = 0;
    }

    private Dictionary<int, LoadedChunk> m_loaded_chunks = new Dictionary<int, LoadedChunk>();
    private Dictionary<int, LoadedChunkHeightData> m_loaded_chunk_height_data = new Dictionary<int, LoadedChunkHeightData>();

    private int m_chunk_id_last = -1;

    private static ProfilerMarker s_zone_transition_marker = new ProfilerMarker("WorldStreaming.HandleZoneTransition");
    private static ProfilerMarker s_zone_finalization_marker = new ProfilerMarker("WorldStreaming.HandleZoneFinalization");
    private static ProfilerMarker s_lod_marker = new ProfilerMarker("WorldStreaming.SelectActiveLod");

    public World m_world;
    public GameObject m_tracking_object;
    public Material m_terrain_material;
    public bool m_draw_lod = false;

    public void Start()
    {
        Assert.IsNotNull(m_world);
        Assert.IsNotNull(m_tracking_object);
        Assert.IsNotNull(m_terrain_material);

        Assert.IsTrue(LOD_SWITCH_DISTANCE.Length == LOD_COUNT);
        Assert.IsTrue(LOD_DISTANCE_PER_VERT.Length == LOD_COUNT);
        Assert.IsTrue(LOD_DEBUG_COLOR.Length == LOD_COUNT);
    }

    public void Update()
    {
        Vector3 position = m_tracking_object.transform.position;

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
            foreach (KeyValuePair<int, LoadedChunk> kvp in m_loaded_chunks)
            {
                LoadedChunk chunk = kvp.Value;
                if (chunk.load_finalized && Time.frameCount % LOD_UPDATE_SLICES == chunk.lod_frame_idx)
                {
                    int lod = SelectActiveLod(position, GetChunkCoords(kvp.Key));
                    if (lod != chunk.lod_active_idx)
                    {
                        chunk.lod_active_idx = lod;
                        chunk.obj.SetActive(lod != -1);

                        for (int i = 0; i < chunk.lod_info.Length; ++i)
                        {
                            chunk.lod_info[i].obj.SetActive(i == lod);
                        }
                    }
                }
            }
        }
    }

    public void OnDestroy()
    {
        foreach (KeyValuePair<int, LoadedChunkHeightData> height_data in m_loaded_chunk_height_data)
        {
            FreeChunkHeightDataResources(height_data.Value);
        }
    }

    private const int CHUNKS_PER_ROW = (int)WorldGen.MAX_WORLD_COORDINATE * 2 / WorldChunk.SIZE;

    public static int GetChunkId(float x, float z)
    {
        x += WorldGen.MAX_WORLD_COORDINATE;
        z += WorldGen.MAX_WORLD_COORDINATE;

        int x_grid = (int)x / WorldChunk.SIZE;
        int z_grid = (int)z / WorldChunk.SIZE;

        return x_grid + z_grid * CHUNKS_PER_ROW;
    }

    public static Vector2 GetChunkCoords(int chunk_id)
    {
        float x = -WorldGen.MAX_WORLD_COORDINATE;
        float z = -WorldGen.MAX_WORLD_COORDINATE;

        x += (chunk_id % CHUNKS_PER_ROW) * WorldChunk.SIZE;
        z += (chunk_id / CHUNKS_PER_ROW) * WorldChunk.SIZE;

        return new Vector2(x, z);
    }

    private HashSet<int> GetTargetChunks(Vector3 position)
    {
        float furthest_radius = LOD_SWITCH_DISTANCE[LOD_SWITCH_DISTANCE.Length - 1];

        Vector2 top_left = new Vector2(position.x - furthest_radius - WorldChunk.SIZE, position.z - furthest_radius - WorldChunk.SIZE);
        Vector2 bottom_right = new Vector2(position.x + furthest_radius + WorldChunk.SIZE, position.z + furthest_radius + WorldChunk.SIZE);

        // align to chunk boundaries
        top_left.x = WorldChunk.SIZE * (int)(top_left.x / WorldChunk.SIZE);
        top_left.y = WorldChunk.SIZE * (int)(top_left.y / WorldChunk.SIZE);

        bottom_right.x = WorldChunk.SIZE * (int)(bottom_right.x / WorldChunk.SIZE);
        bottom_right.y = WorldChunk.SIZE * (int)(bottom_right.y / WorldChunk.SIZE);

        Assert.IsTrue(top_left.x % WorldChunk.SIZE == 0.0f);
        Assert.IsTrue(top_left.y % WorldChunk.SIZE == 0.0f);
        Assert.IsTrue(bottom_right.x % WorldChunk.SIZE == 0.0f);
        Assert.IsTrue(bottom_right.x % WorldChunk.SIZE == 0.0f);

        HashSet<int> chunks = new HashSet<int>();

        for (float z = top_left.y; z < bottom_right.y; z += WorldChunk.SIZE)
        {
            for (float x = top_left.x; x < bottom_right.x; x += WorldChunk.SIZE)
            {
                chunks.Add(GetChunkId(x, z));
            }
        }

        return chunks;
    }

    private void FreeChunkMeshCreationResources(LoadedChunk chunk)
    {
        foreach (LoadedChunkMeshParams param in chunk.lod_mesh_params)
        {
            param.verts.Dispose();
            param.tris.Dispose();
            param.uvs.Dispose();
            param.normals.Dispose();
        }

        chunk.lod_mesh_params = null;
        chunk.nearby_heights = null;
    }

    private void FreeChunkHeightDataResources(LoadedChunkHeightData data)
    {
        data.heights.Dispose();
    }

    private void RemoveNeighbouringRefCount(int start_chunk_id)
    {
        Vector2 position = GetChunkCoords(start_chunk_id);

        for (int z = -1; z <= 1; ++z)
        {
            for (int x = -1; x <= 1; ++x)
            {
                float chunk_x = position.x + x * WorldChunk.SIZE;
                float chunk_z = position.y + z * WorldChunk.SIZE;
                int chunk_id = GetChunkId(chunk_x, chunk_z);

                LoadedChunkHeightData height_data;
                if (m_loaded_chunk_height_data.TryGetValue(chunk_id, out height_data))
                {
                    if (--height_data.ref_count == 0)
                    {
                        height_data.heights.Dispose();
                        m_loaded_chunk_height_data.Remove(chunk_id);
                    }
                }
            }
        }
    }

    private int SelectActiveLod(Vector3 pos, Vector2 chunk_coords)
    {
        chunk_coords.x += WorldChunk.SIZE / 2.0f;
        chunk_coords.y += WorldChunk.SIZE / 2.0f;

        float dist_squared = (new Vector2(pos.x, pos.z) - chunk_coords).sqrMagnitude;

        int selected_lod_idx = -1;

        for (int i = 0; i < LOD_SWITCH_DISTANCE.Length; ++i)
        {
            float lod_switch_squared = LOD_SWITCH_DISTANCE[i] * LOD_SWITCH_DISTANCE[i];

            if (dist_squared < lod_switch_squared)
            {
                selected_lod_idx = i;
                break;
            }
        }

        return selected_lod_idx;
    }

    private void HandleZoneTransition(Vector3 position)
    {
        HashSet<int> chunks = GetTargetChunks(position);

        foreach (int chunk in chunks)
        {
            if (!m_loaded_chunks.ContainsKey(chunk))
            {
                m_loaded_chunks.Add(chunk, new LoadedChunk());
            }
        }

        foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
        {
            if (!chunks.Contains(chunk.Key))
            {
                chunk.Value.unload = true;
            }
        }
    }

    private void AddChunkRendering(GameObject obj, int lod, Mesh mesh)
    {
        MeshRenderer meshRenderer = obj.AddComponent<MeshRenderer>();
        meshRenderer.sharedMaterial = m_terrain_material;

        if (m_draw_lod)
        {
            meshRenderer.material.SetColor("_BaseColor", LOD_DEBUG_COLOR[lod]);
        }

        MeshFilter meshFilter = obj.AddComponent<MeshFilter>();
        meshFilter.mesh = mesh;
    }

    private void HandleZoneFinalization(float timeslice)
    {
        float time_start = Time.realtimeSinceStartup;

        List<int> unloaded_chunks = new List<int>();

        foreach (KeyValuePair<int, LoadedChunk> kvp in m_loaded_chunks)
        {
            LoadedChunk chunk = kvp.Value;
            if (!chunk.load_started)
            {
                chunk.load_started = true;
                chunk.nearby_heights = new NativeArray<float>[9];

                Vector2 position = GetChunkCoords(kvp.Key);
                NativeArray<JobHandle> chunk_gen_handles = new NativeArray<JobHandle>(9, Allocator.Temp);

                int i = 0;
                for (int z = -1; z <= 1; ++z)
                {
                    for (int x = -1; x <= 1; ++x)
                    {
                        float chunk_x = position.x + x * WorldChunk.SIZE;
                        float chunk_z = position.y + z * WorldChunk.SIZE;
                        int chunk_id = GetChunkId(chunk_x, chunk_z);

                        LoadedChunkHeightData height_data;

                        if (!m_loaded_chunk_height_data.TryGetValue(chunk_id, out height_data))
                        {
                            height_data = new LoadedChunkHeightData();
                            height_data.job = m_world.gen.ScheduleBaseTerrainGeneration((int)chunk_x, (int)chunk_z, WorldChunk.SIZE_WITH_BORDER, out height_data.heights);
                            m_loaded_chunk_height_data.Add(chunk_id, height_data);
                        }

                        chunk_gen_handles[i] = height_data.job;
                        chunk.nearby_heights[i++] = height_data.heights;
                        ++height_data.ref_count;
                    }
                }

                JobHandle chunk_gen_handle = JobHandle.CombineDependencies(chunk_gen_handles);
                chunk_gen_handles.Dispose();

                NativeArray<JobHandle> mesh_gen_handles = new NativeArray<JobHandle>(LOD_COUNT, Allocator.Temp);

                chunk.lod_mesh_params = new LoadedChunkMeshParams[LOD_COUNT];

                for (int lod = 0; lod < LOD_COUNT; ++lod)
                {
                    LoadedChunkMeshParams param = new LoadedChunkMeshParams();
                    mesh_gen_handles[lod] = WorldChunk.ScheduleChunkMeshGeneration(
                        chunk.nearby_heights, LOD_DISTANCE_PER_VERT[lod],
                        out param.verts, out param.tris, out param.uvs, out param.normals,
                        out param.final_vert_count, out param.final_tri_count, chunk_gen_handle);
                    chunk.lod_mesh_params[lod] = param;
                }

                chunk.load_job = JobHandle.CombineDependencies(mesh_gen_handles);
                mesh_gen_handles.Dispose();
            }
            else if (chunk.unload)
            {
                if (chunk.load_finalized) // finalized, we only need to free the objects
                {
                    Destroy(chunk.obj);
                    foreach (LodInfo info in chunk.lod_info)
                    {
                        Destroy(info.obj);
                    }
                    unloaded_chunks.Add(kvp.Key);
                }
                else if (chunk.load_job.IsCompleted) // not finalized, but tasks done - we can release resources
                {
                    chunk.load_job.Complete();
                    FreeChunkMeshCreationResources(chunk);
                    unloaded_chunks.Add(kvp.Key);
                }
            }
            else if (!chunk.load_finalized && chunk.load_job.IsCompleted)
            {
                chunk.load_job.Complete();
                chunk.load_finalized = true;

                Vector2 chunk_pos = GetChunkCoords(kvp.Key);

                GameObject obj = new GameObject(string.Format("Chunk{0}", kvp.Key));
                obj.transform.parent = m_world.gameObject.transform;
                obj.transform.position = new Vector3(chunk_pos.x, 0.0f, chunk_pos.y);
                obj.SetActive(false);
                chunk.obj = obj;

                chunk.lod_info = new LodInfo[LOD_COUNT];

                for (int i = 0; i < LOD_COUNT; ++i)
                {
                    LoadedChunkMeshParams param = chunk.lod_mesh_params[i];
                    Assert.IsTrue(chunk.lod_mesh_params.Length == LOD_COUNT);

                    GameObject lod_obj = new GameObject(string.Format("Lod{1}", kvp.Key, i));
                    lod_obj.transform.parent = obj.transform;
                    lod_obj.transform.localPosition = Vector3.zero;
                    Mesh lod_mesh = WorldChunk.FinalizeChunkMesh(param.verts, param.tris, param.uvs, param.normals,
                        param.final_vert_count, param.final_tri_count);
                    AddChunkRendering(lod_obj, i, lod_mesh);

                    GameObject skirt_obj = new GameObject("Skirt");
                    skirt_obj.transform.parent = lod_obj.transform;
                    skirt_obj.transform.localPosition = Vector3.zero;
                    AddChunkRendering(skirt_obj, i, WorldChunk.CreateChunkSkirting(lod_mesh.vertices, lod_mesh.normals));

                    if (Debug.isDebugBuild)
                    {
                        lod_obj.AddComponent<MeshNormalRenderer>();
                    }

                    chunk.lod_info[i] = new LodInfo();
                    chunk.lod_info[i].border_normals = WorldChunk.ExtractBorderNormals(param.normals);
                    chunk.lod_info[i].obj = lod_obj;
                    chunk.lod_info[i].skirts = skirt_obj;
                }

                chunk.lod_frame_idx = Time.frameCount % LOD_UPDATE_SLICES;

                FreeChunkMeshCreationResources(chunk);
            }

            if (Time.realtimeSinceStartup - time_start > timeslice)
            {
                break;
            }
        }

        foreach (int chunk in unloaded_chunks)
        {
            RemoveNeighbouringRefCount(chunk);
            m_loaded_chunks.Remove(chunk);
        }
    }
}
