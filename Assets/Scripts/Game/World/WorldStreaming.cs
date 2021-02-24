using System.Collections.Generic;
using Unity.Collections;
using Unity.Jobs;
using Unity.Profiling;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldStreaming : MonoBehaviour
{
    private const int LOD_COUNT = 1;

    private static readonly float[] LOD_SWITCH_DISTANCE =
    {
        128.0f,
        //256.0f,
        //512.0f,
        //1024.0f
    };

    private static readonly int[] LOD_DISTANCE_PER_VERT =
    {
        1,
        //4,
        //16,
        //WorldChunk.SIZE / 2
    };

    private static readonly Color[] LOD_DEBUG_COLOR =
    {
        Color.red,
        //new Color(1.0f, 0.7f, 0.0f),
        //Color.yellow,
        //Color.green
    };

    private const int LOD_UPDATE_SLICES = LOD_COUNT;

    private class LoadedChunkMeshParams
    {
        public NativeArray<Vector3> verts;
        public NativeArray<int> tris;
        public NativeArray<Vector2> uvs;
        public NativeArray<Vector3> normals;
    }

    private class LodInfo
    {
        public GameObject obj;
        public GameObject skirts;
        public Vector3[] border_normals;
    }

    private class LoadedChunk
    {
        public int id;
        public bool unload = false;

        public GameObject obj;

        public LodInfo[] lod_info;
        public int lod_frame_idx = -1;
        public int lod_active_idx = -2;
        public LoadedChunkMeshParams[] lod_mesh_params;

        public NativeArray<float> heights;

        public bool load_started = false;
        public bool load_finalized = false;
        public JobHandle load_job;
    }

    private Dictionary<int, LoadedChunk> m_loaded_chunks = new Dictionary<int, LoadedChunk>();

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
                    int lod = SelectActiveLod(position, chunk);
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

    private JobHandle ScheduleChunkLoad(int chunk_id, out LoadedChunkMeshParams[] mesh_params, out NativeArray<float> heights)
    {
        Vector2 position = GetChunkCoords(chunk_id);
        mesh_params = new LoadedChunkMeshParams[LOD_COUNT];
        JobHandle chunk_gen_handle = WorldChunk.ScheduleChunkGeneration((int)position.x, (int)position.y, m_world.gen.GetPerlin(), out heights);

        for (int i = 0; i < LOD_COUNT; ++i)
        {
            mesh_params[i] = new LoadedChunkMeshParams();
            JobHandle mesh_handle = WorldChunk.ScheduleChunkMeshGeneration(heights, LOD_DISTANCE_PER_VERT[i],
                out mesh_params[i].verts, out mesh_params[i].tris, out mesh_params[i].uvs, out mesh_params[i].normals,
                chunk_gen_handle);
            chunk_gen_handle = JobHandle.CombineDependencies(chunk_gen_handle, mesh_handle);
        }

        return chunk_gen_handle;
    }

    private void FreeChunkResources(LoadedChunk chunk)
    {
        foreach (LoadedChunkMeshParams param in chunk.lod_mesh_params)
        {
            param.verts.Dispose();
            param.tris.Dispose();
            param.uvs.Dispose();
            param.normals.Dispose();
        }

        chunk.heights.Dispose();
    }

    private int SelectActiveLod(Vector3 pos, LoadedChunk chunk)
    {
        Vector2 chunk_coords = GetChunkCoords(chunk.id);
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

        int base_y = (0 / WorldChunk.SIZE) * WorldChunk.SIZE;

        chunks.Clear();
        chunks.Add(GetChunkId(0, base_y));
        chunks.Add(GetChunkId(0, base_y + WorldChunk.SIZE));

        foreach (int chunk in chunks)
        {
            if (!m_loaded_chunks.ContainsKey(chunk))
            {
                m_loaded_chunks.Add(chunk, new LoadedChunk { id = chunk });
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

        foreach (KeyValuePair<int, LoadedChunk> chunk in m_loaded_chunks)
        {
            if (!chunk.Value.load_started)
            {
                chunk.Value.load_started = true;
                chunk.Value.load_job = ScheduleChunkLoad(chunk.Key, out chunk.Value.lod_mesh_params, out chunk.Value.heights);
            }
            else if (chunk.Value.unload)
            {
                if (chunk.Value.load_finalized) // finalized, we only need to free the objects
                {
                    Object.Destroy(chunk.Value.obj);
                    foreach (LodInfo info in chunk.Value.lod_info)
                    {
                        Object.Destroy(info.obj);
                    }
                    unloaded_chunks.Add(chunk.Key);
                }
                else if (chunk.Value.load_job.IsCompleted) // not finalized, but tasks done - we can release resources
                {
                    chunk.Value.load_job.Complete();
                    FreeChunkResources(chunk.Value);
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
                obj.SetActive(false);
                chunk.Value.obj = obj;

                chunk.Value.lod_info = new LodInfo[LOD_COUNT];

                for (int i = 0; i < LOD_COUNT; ++i)
                {
                    LoadedChunkMeshParams param = chunk.Value.lod_mesh_params[i];
                    Assert.IsTrue(chunk.Value.lod_mesh_params.Length == LOD_COUNT);

                    GameObject lod_obj = new GameObject(string.Format("Lod{1}", chunk.Key, i));
                    lod_obj.transform.parent = obj.transform;
                    lod_obj.transform.localPosition = Vector3.zero;
                    Mesh lod_mesh = WorldChunk.FinalizeChunkMesh(param.verts, param.tris, param.uvs, param.normals);
                    AddChunkRendering(lod_obj, i, lod_mesh);

                    GameObject skirt_obj = new GameObject("Skirt");
                    skirt_obj.transform.parent = lod_obj.transform;
                    skirt_obj.transform.localPosition = Vector3.zero;
                    AddChunkRendering(skirt_obj, i, WorldChunk.CreateChunkSkirting(lod_mesh.vertices, lod_mesh.normals));

                    if (Debug.isDebugBuild)
                    {
                        lod_obj.AddComponent<MeshNormalRenderer>();
                    }

                    chunk.Value.lod_info[i] = new LodInfo();
                    chunk.Value.lod_info[i].border_normals = WorldChunk.ExtractBorderNormals(param.normals);
                    chunk.Value.lod_info[i].obj = lod_obj;
                    chunk.Value.lod_info[i].skirts = skirt_obj;
                }

                chunk.Value.lod_frame_idx = Time.frameCount % LOD_UPDATE_SLICES;

                FreeChunkResources(chunk.Value);
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
