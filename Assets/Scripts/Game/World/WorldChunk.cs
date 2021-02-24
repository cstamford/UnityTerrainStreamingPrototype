using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldChunk
{
    public const int SIZE = 64;

    private const int SIZE_WITH_BORDER = SIZE + 1;
    private const int TRIM = 2;

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

    public static JobHandle ScheduleChunkGeneration(int chunk_x, int chunk_z, PerlinGenerator perlin, out NativeArray<float> heights)
    {
        int heights_count = (SIZE_WITH_BORDER + TRIM) * (SIZE_WITH_BORDER + TRIM); // always calc full precision heights even if they aren't used
        NativeArray<Vector2> coords = new NativeArray<Vector2>(heights_count, Allocator.TempJob);

        int i = 0;
        for (int z = chunk_z - 1; z < chunk_z + SIZE_WITH_BORDER + 1; ++z)
        {
            for (int x = chunk_x - 1; x < chunk_x + SIZE_WITH_BORDER + 1; ++x)
            {
                coords[i++] = new Vector2(x, z);
            }
        }

        heights = new NativeArray<float>(coords.Length, Allocator.Persistent);

        GenerateChunkJob job = new GenerateChunkJob()
        {
            coords = coords,
            perlin = perlin,
            heights = heights
        };

        return job.Schedule(coords.Length, 32);
    }

    [BurstCompile]
    private struct GenerateChunkMeshJob : IJob
    {
        [ReadOnly] public NativeArray<float> heights;
        [ReadOnly] public int distance_per_vert;

        public NativeArray<Vector3> verts;
        public NativeArray<Vector2> uvs;
        public NativeArray<int> tris;

        public void Execute()
        {
            // -- Step 1:
            // Generate mesh with subdivision.
            // This is our main terrain that will be visible in the game.

            int vert_counter = 0;
            for (int z = 0; z < SIZE_WITH_BORDER; z += distance_per_vert)
            {
                for (int x = 0; x < SIZE_WITH_BORDER; x += distance_per_vert)
                {
                    int height_idx = (x + 1) + (z + 1) * (SIZE_WITH_BORDER + TRIM);
                    verts[vert_counter] = new Vector3(x, heights[height_idx], z);
                    uvs[vert_counter] = new Vector2((float)x / SIZE_WITH_BORDER, (float)z / SIZE_WITH_BORDER);
                    ++vert_counter;
                }
            }

            int stride = SIZE / distance_per_vert;

            int tri_counter = 0;
            for (int z = 0; z < stride; ++z)
            {
                for (int x = 0; x < stride; ++x)
                {
                    int vert_stride = stride + 1;
                    int vert_idx = x + z * vert_stride;
                    int vert_idx_below = vert_idx + vert_stride;
                    MakeFace(vert_idx, vert_idx + 1, vert_idx_below + 1, vert_idx_below, tris, ref tri_counter);
                }
            }
        }
    }

    [BurstCompile]
    private struct GenerateChunkNormalsJob : IJob
    {
        [ReadOnly] public NativeSlice<Vector3> verts;
        [ReadOnly] public NativeArray<int> tris;

        public NativeArray<Vector3> normals;

        public void Execute()
        {
            // -- Step 3:
            // Calculate normals to prevent seams between neighbouring chunks.

            for (int tri = 0; tri < tris.Length; tri += 3)
            {
                Vector3 t0v = verts[tris[tri + 0]];
                Vector3 t1v = verts[tris[tri + 1]];
                Vector3 t2v = verts[tris[tri + 2]];

                Vector3 n = Vector3.Cross(t1v - t0v, t2v - t0v);

                // TODO: I'm not so sure about this code, but it seems to work decently...
                // See https://stackoverflow.com/questions/45477806/general-method-for-calculating-smooth-vertex-normals-with-100-smoothness

                float a0 = Vector3.Angle(t1v - t0v, t2v - t0v);
                float a1 = Vector3.Angle(t2v - t1v, t0v - t1v);
                float a2 = Vector3.Angle(t0v - t2v, t1v - t2v);

                Vector3 n0 = n * a0;
                Vector3 n1 = n * a1;
                Vector3 n2 = n * a2;

                for (int i = 0; i < 3; ++i)
                {
                    int v = tris[tri + i];
                    normals[v] += n0;
                    normals[v] += n1;
                    normals[v] += n2;
                }
            }

            for (int i = 0; i < normals.Length; ++i)
            {
                normals[i] = Vector3.Normalize(normals[i]);
            }
        }
    }

    public static JobHandle ScheduleChunkMeshGeneration(NativeArray<float> heights, int distance_per_vert,
        out NativeArray<Vector3> verts, out NativeArray<int> tris, out NativeArray<Vector2> uvs, out NativeArray<Vector3> normals,
        out int final_vert_count, out int final_tri_count, JobHandle depends = default)
    {
        ChunkMeshLayoutInfo info = CalculateChunkMeshLayoutInfo(distance_per_vert, true);

        verts = new NativeArray<Vector3>(info.vert_count, Allocator.Persistent);
        uvs = new NativeArray<Vector2>(info.vert_count, Allocator.Persistent);
        tris = new NativeArray<int>(info.tri_count, Allocator.Persistent);

        GenerateChunkMeshJob generate_mesh_job = new GenerateChunkMeshJob()
        {
            heights = heights,
            distance_per_vert = distance_per_vert,
            verts = verts,
            uvs = uvs,
            tris = tris
        };

        JobHandle generate_mesh_job_handle = generate_mesh_job.Schedule(depends);

        normals = new NativeArray<Vector3>(info.vert_count, Allocator.Persistent);

        GenerateChunkNormalsJob generate_chunk_normals_job = new GenerateChunkNormalsJob()
        {
            verts = verts,
            tris = tris,
            normals = normals
        };

        JobHandle generate_chunk_normals_job_handle = generate_chunk_normals_job.Schedule(generate_mesh_job_handle);

        ChunkMeshLayoutInfo info_without_trim = CalculateChunkMeshLayoutInfo(distance_per_vert, false);
        final_vert_count = info_without_trim.vert_count;
        final_tri_count = info_without_trim.tri_count;

        return generate_chunk_normals_job_handle;
    }

    public static Mesh FinalizeChunkMesh(NativeArray<Vector3> verts, NativeArray<int> tris, NativeArray<Vector2> uvs, NativeArray<Vector3> normals,
        int final_vert_count, int final_tri_count)
    {
        Mesh mesh = new Mesh();
        mesh.SetVertices(verts, 0, final_vert_count);
        mesh.SetUVs(0, uvs, 0, final_vert_count);
        mesh.SetNormals(normals, 0, final_vert_count);
        mesh.SetTriangles(tris.Slice(0, final_tri_count).ToArray(), 0);
        return mesh;
    }

    public static Vector3[] ExtractBorderNormals(NativeArray<Vector3> normals)
    {
        int stride = (int)Mathf.Sqrt(normals.Length);

        Vector3[] norms = new Vector3[(stride - 1) * 4];
        int norm_counter = 0;

        // -- Top row
        for (int x = 0; x < stride; ++x)
        {
            norms[norm_counter++] = normals[x];
        }

        // -- Bottom row
        for (int x = 0; x < stride; ++x)
        {
            norms[norm_counter++] = normals[x + (stride - 1) * stride];
        }

        // -- Left column
        for (int z = 1; z < stride - 1; ++z)
        {
            norms[norm_counter++] = normals[z * stride];
        }

        // -- Right column
        for (int z = 1; z < stride - 1; ++z)
        {
            norms[norm_counter++] = normals[stride + z * stride];
        }

        return norms;
    }

    private static void CreateChunkVertColumn(Vector3 v, Vector3 n, float u, ref int vert_counter,
        NativeArray<Vector3> verts, NativeArray<Vector2> uvs, NativeArray<Vector3> normals)
    {
        verts[vert_counter] = new Vector3(v.x, v.y, v.z);
        uvs[vert_counter] = new Vector2(u, 0.0f);
        normals[vert_counter++] = n;

        verts[vert_counter] = new Vector3(v.x, v.y - 5.0f, v.z);
        uvs[vert_counter] = new Vector2(u, 1.0f);
        normals[vert_counter++] = n;
    }

    public static Mesh CreateChunkSkirting(Vector3[] chunk_verts, Vector3[] chunk_norms)
    {
        int stride_with_border = (int)Mathf.Sqrt(chunk_verts.Length);
        int stride = stride_with_border - 1;

        NativeArray<Vector3> verts = new NativeArray<Vector3>(stride_with_border * 2 * 4, Allocator.Temp);
        NativeArray<Vector2> uvs = new NativeArray<Vector2>(verts.Length, Allocator.Temp);
        NativeArray<Vector3> normals = new NativeArray<Vector3>(verts.Length, Allocator.Temp);
        NativeArray<int> tris = new NativeArray<int>(stride * 6 * 4, Allocator.Temp);

        const int TOP_LEFT = 0;
        const int TOP_RIGHT = 2;
        const int BOTTOM_RIGHT = 3;
        const int BOTTOM_LEFT = 1;

        int vert_counter = 0;
        int tri_counter = 0;

        // -- Top row
        int face_counter = 0;
        for (int x = 0; x < stride_with_border; ++x)
        {
            int v_idx = x;
            CreateChunkVertColumn(chunk_verts[v_idx], chunk_norms[v_idx], x / stride_with_border, ref vert_counter, verts, uvs, normals);

            if (x < stride)
            {
                int idx = face_counter++ * 2;
                MakeFace(idx + BOTTOM_LEFT, idx + BOTTOM_RIGHT, idx + TOP_RIGHT, idx + TOP_LEFT, tris, ref tri_counter);
            }
        }

        // -- Bottom row
        ++face_counter;
        for (int x = 0; x < stride_with_border; ++x)
        {
            int v_idx = x + (stride_with_border - 1) * stride_with_border;
            CreateChunkVertColumn(chunk_verts[v_idx], chunk_norms[v_idx], x / stride_with_border, ref vert_counter, verts, uvs, normals);

            if (x < stride)
            {
                int idx = face_counter++ * 2;
                MakeFace(idx + TOP_LEFT, idx + TOP_RIGHT, idx + BOTTOM_RIGHT, idx + BOTTOM_LEFT, tris, ref tri_counter);
            }
        }

        // -- Left column
        ++face_counter;
        for (int z = 0; z < stride_with_border; ++z)
        {
            int v_idx = z * stride_with_border;
            CreateChunkVertColumn(chunk_verts[v_idx], chunk_norms[v_idx], z / stride_with_border, ref vert_counter, verts, uvs, normals);

            if (z < stride)
            {
                int idx = face_counter++ * 2;
                MakeFace(idx + TOP_LEFT, idx + TOP_RIGHT, idx + BOTTOM_RIGHT, idx + BOTTOM_LEFT, tris, ref tri_counter);
            }
        }

        // -- Right column
        ++face_counter;
        for (int z = 0; z < stride_with_border; ++z)
        {
            int v_idx = (stride_with_border - 1) + z * stride_with_border;
            CreateChunkVertColumn(chunk_verts[v_idx], chunk_norms[v_idx], z / stride_with_border, ref vert_counter, verts, uvs, normals);;

            if (z < stride)
            {
                int idx = face_counter++ * 2;
                MakeFace(idx + BOTTOM_LEFT, idx + BOTTOM_RIGHT, idx + TOP_RIGHT, idx + TOP_LEFT, tris, ref tri_counter);
            }
        }

        Assert.IsTrue(vert_counter == verts.Length);
        Assert.IsTrue(tri_counter == tris.Length);

        Mesh mesh = new Mesh();
        mesh.SetVertices(verts);
        mesh.SetUVs(0, uvs);
        mesh.SetNormals(normals);
        mesh.SetTriangles(tris.ToArray(), 0);
        return mesh;
    }

    private struct ChunkMeshLayoutInfo
    {
        public int vert_count;
        public int tri_count;
    }

    private static ChunkMeshLayoutInfo CalculateChunkMeshLayoutInfo(int distance_per_vert, bool trim)
    {
        int vert_count_one_row = Mathf.CeilToInt(SIZE_WITH_BORDER / (float)distance_per_vert);

        if (trim)
        {
            vert_count_one_row += TRIM;
        }

        int vert_count = vert_count_one_row * vert_count_one_row;
        int tri_count = (vert_count_one_row - 1) * (vert_count_one_row - 1) * 6;
        return new ChunkMeshLayoutInfo() { vert_count = vert_count, tri_count = tri_count };
    }

    private static void MakeFace(int v0, int v1, int v2, int v3, NativeArray<int> tris, ref int tri_counter)
    {
        tris[tri_counter++] = v0;
        tris[tri_counter++] = v3;
        tris[tri_counter++] = v1;
        tris[tri_counter++] = v1;
        tris[tri_counter++] = v3;
        tris[tri_counter++] = v2;
    }
}
