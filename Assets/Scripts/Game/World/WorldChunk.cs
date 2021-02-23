using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldChunk
{
    public const int SIZE = 64;

    private const int SIZE_WITH_BORDER = SIZE + 1;

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
        int heights_count = SIZE_WITH_BORDER * SIZE_WITH_BORDER; // always calc full precision heights even if they aren't used
        NativeArray<Vector2> coords = new NativeArray<Vector2>(heights_count, Allocator.TempJob);

        int i = 0;
        for (int z = chunk_z; z < chunk_z + SIZE_WITH_BORDER; ++z)
        {
            for (int x = chunk_x; x < chunk_x + SIZE_WITH_BORDER; ++x)
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
                    int height_idx = x + z * SIZE_WITH_BORDER;
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

            Assert.IsTrue(vert_counter == verts.Length);
            Assert.IsTrue(tri_counter == tris.Length);
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
                normals[i].Normalize();
            }
        }
    }

    public static JobHandle ScheduleChunkMeshGeneration(NativeArray<float> heights, int distance_per_vert,
        out NativeArray<Vector3> verts, out NativeArray<int> tris,  out NativeArray<Vector2> uvs, out NativeArray<Vector3> normals,
        JobHandle depends = default)
    {
        ChunkMeshLayoutInfo info = CalculateChunkMeshLayoutInfo(distance_per_vert);

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
        return generate_chunk_normals_job_handle;
    }

    public static Mesh FinalizeChunkMesh(NativeArray<Vector3> verts, NativeArray<int> tris, NativeArray<Vector2> uvs, NativeArray<Vector3> normals)
    {
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

    private static ChunkMeshLayoutInfo CalculateChunkMeshLayoutInfo(int distance_per_vert)
    {
        int vert_count_one_row = Mathf.CeilToInt(SIZE_WITH_BORDER / (float)distance_per_vert);
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
