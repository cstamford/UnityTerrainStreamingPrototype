using Unity.Burst;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;
using UnityEngine.Assertions;

public class WorldChunk
{
    public enum Lod
    {
        FullQuality = 1,
        HalfQuality = 2,
        QuarterQuality = 4
    }

    public const int CHUNK_SIZE = 64;
    private const int CHUNK_SIZE_WITH_BORDER = CHUNK_SIZE + 1;
    private const int CHUNK_SIZE_WITH_NORMAL_BLENDING = CHUNK_SIZE_WITH_BORDER + 2;

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
                perlin.GetHeightAt(coords[id].x / 16.0f, coords[id].y / 16.0f) +
                Mathf.Max(
                    perlin.GetHeightAt((coords[id].x + 250.0f) / 4.0f, (coords[id].y + 250.0f) / 4.0f),
                    perlin.GetHeightAt((coords[id].x - 100.0f) / 4.0f, (coords[id].y - 100.0f) / 4.0f));
        }
    }

    public static JobHandle ScheduleChunkGeneration(int chunk_x, int chunk_z, Lod lod, PerlinGenerator perlin, out NativeArray<float> heights)
    {
        NativeArray<Vector2> coords = new NativeArray<Vector2>(CHUNK_SIZE_WITH_NORMAL_BLENDING * CHUNK_SIZE_WITH_NORMAL_BLENDING, Allocator.Persistent);

        int i = 0;
        for (int z = chunk_z; z < chunk_z + CHUNK_SIZE_WITH_NORMAL_BLENDING; ++z)
        {
            for (int x = chunk_x; x < chunk_x + CHUNK_SIZE_WITH_NORMAL_BLENDING; ++x)
            {
                coords[i++] = new Vector2(x - 1, z - 1);
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
        [DeallocateOnJobCompletion]
        [ReadOnly] public NativeArray<float> heights;

        public NativeArray<Vector3> verts;
        public NativeArray<Vector2> uvs;
        public NativeArray<int> tris;

        public void Execute()
        {
            // -- Step 1:
            // Generate mesh with:
            //      verts: CHUNK_SIZE_WITH_BORDER x CHUNK_SIZE_WITH_BORDER
            //      tris: CHUNK_SIZE * CHUNK_SIZE * 2
            // This is our main terrain that will be visible in the game.

            int vert_counter = 0;
            for (int z = 0; z < CHUNK_SIZE_WITH_BORDER; ++z)
            {
                for (int x = 0; x < CHUNK_SIZE_WITH_BORDER; ++x)
                {
                    int height_idx = (x + 1) + (z + 1) * CHUNK_SIZE_WITH_NORMAL_BLENDING;
                    verts[vert_counter] = new Vector3(x, heights[height_idx], z);
                    uvs[vert_counter] = new Vector2(x, z);
                    ++vert_counter;
                }
            }

            int tri_counter = 0;
            for (int z = 0; z < CHUNK_SIZE; ++z)
            {
                for (int x = 0; x < CHUNK_SIZE; ++x)
                {
                    int vert_idx = x + z * CHUNK_SIZE_WITH_BORDER;
                    int vert_idx_below = vert_idx + CHUNK_SIZE_WITH_BORDER;
                    MakeFace(vert_idx, vert_idx + 1, vert_idx_below + 1, vert_idx_below, tris, ref tri_counter);
                }
            }

            // -- Step 2:
            // Append to this mesh verts for the outside border + tris for them.
            // You can think of this as "tape" around the mesh to assist in calculating normals.

            DoCorners(verts, ref vert_counter, tris, ref tri_counter, heights,
                DoTopRow(verts, ref vert_counter, tris, ref tri_counter, heights),
                DoBottomRow(verts, ref vert_counter, tris, ref tri_counter, heights),
                DoLeftColumn(verts, ref vert_counter, tris, ref tri_counter, heights),
                DoRightColumn(verts, ref vert_counter, tris, ref tri_counter, heights));

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
                    if (v < normals.Length) // exclude tape verts
                    {
                        normals[v] += n0;
                        normals[v] += n1;
                        normals[v] += n2;
                    }
                }
            }

            for (int i = 0; i < normals.Length; ++i)
            {
                normals[i].Normalize();
            }
        }
    }

    public static JobHandle ScheduleChunkMeshGeneration(NativeArray<float> heights,
        out NativeArray<Vector3> verts, out NativeArray<int> tris,  out NativeArray<Vector2> uvs, out NativeArray<Vector3> normals,
        JobHandle depends = default)
    {
        verts = new NativeArray<Vector3>(CHUNK_SIZE_WITH_NORMAL_BLENDING * CHUNK_SIZE_WITH_NORMAL_BLENDING, Allocator.Persistent);
        uvs = new NativeArray<Vector2>(CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER, Allocator.Persistent);
        tris = new NativeArray<int>((CHUNK_SIZE_WITH_NORMAL_BLENDING - 1) * (CHUNK_SIZE_WITH_NORMAL_BLENDING - 1) * 2 * 3, Allocator.Persistent);

        GenerateChunkMeshJob generate_mesh_job = new GenerateChunkMeshJob()
        {
            heights = heights,
            verts = verts,
            uvs = uvs,
            tris = tris
        };

        JobHandle generate_mesh_job_handle = generate_mesh_job.Schedule(depends);

        normals = new NativeArray<Vector3>(CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER, Allocator.Persistent);

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
        // Remove the outside tape, so we're left with the good mesh.
        Mesh mesh = new Mesh();
        mesh.SetVertices(verts, 0, CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER);
        mesh.SetUVs(0, uvs);
        mesh.SetNormals(normals);
        mesh.SetTriangles(tris.Slice(0, CHUNK_SIZE * CHUNK_SIZE * 2 * 3).ToArray(), 0);
        return mesh;
    }

    private static int DoTopRow(NativeArray<Vector3> verts, ref int vert_counter, NativeArray<int> tris, ref int tri_counter, NativeArray<float> heights)
    {
        int top_row_vert_counter = vert_counter;
        int top_row_vert_counter_start = vert_counter;

        for (int x = 0; x < CHUNK_SIZE_WITH_BORDER; ++x)
        {
            int idx = x + 1;
            verts[vert_counter++] = new Vector3(x, heights[idx], -1);
        }

        for (int x = 0; x < CHUNK_SIZE; ++x)
        {
            MakeFace(top_row_vert_counter, top_row_vert_counter + 1, x + 1, x, tris, ref tri_counter);
            ++top_row_vert_counter;
        }

        return top_row_vert_counter_start;
    }

    private static int DoBottomRow(NativeArray<Vector3> verts, ref int vert_counter, NativeArray<int> tris, ref int tri_counter, NativeArray<float> heights)
    {
        int bottom_row_vert_counter = vert_counter;
        int bottom_row_vert_counter_start = vert_counter;

        for (int x = 0; x < CHUNK_SIZE_WITH_BORDER; ++x)
        {
            int idx = (x + 1) + (CHUNK_SIZE_WITH_NORMAL_BLENDING - 1) * CHUNK_SIZE_WITH_NORMAL_BLENDING;
            verts[vert_counter++] = new Vector3(x, heights[idx], CHUNK_SIZE_WITH_BORDER);
        }

        for (int x = 0; x < CHUNK_SIZE; ++x)
        {
            int above_idx = x + CHUNK_SIZE * CHUNK_SIZE_WITH_BORDER;
            MakeFace(above_idx, above_idx + 1, bottom_row_vert_counter + 1, bottom_row_vert_counter, tris, ref tri_counter);
            ++bottom_row_vert_counter;
        }

        return bottom_row_vert_counter_start;
    }

    private static int DoLeftColumn(NativeArray<Vector3> verts, ref int vert_counter, NativeArray<int> tris, ref int tri_counter, NativeArray<float> heights)
    {
        int left_column_vert_counter = vert_counter;
        int left_column_vert_counter_start = vert_counter;

        for (int z = 0; z < CHUNK_SIZE_WITH_BORDER; ++z)
        {
            int idx = (z + 1) * CHUNK_SIZE_WITH_NORMAL_BLENDING;
            verts[vert_counter++] = new Vector3(-1, heights[idx], z);
        }

        for (int z = 0; z < CHUNK_SIZE; ++z)
        {
            int right_idx = z * CHUNK_SIZE_WITH_BORDER;
            int next_right_idx = (z + 1) * CHUNK_SIZE_WITH_BORDER;
            MakeFace(left_column_vert_counter, right_idx, next_right_idx, left_column_vert_counter + 1, tris, ref tri_counter);
            ++left_column_vert_counter;
        }

        return left_column_vert_counter_start;
    }

    private static int DoRightColumn(NativeArray<Vector3> verts, ref int vert_counter, NativeArray<int> tris, ref int tri_counter, NativeArray<float> heights)
    {
        int right_column_vert_counter = vert_counter;
        int right_column_vert_counter_start = right_column_vert_counter;

        for (int z = 0; z < CHUNK_SIZE_WITH_BORDER; ++z)
        {
            int idx = (CHUNK_SIZE_WITH_NORMAL_BLENDING - 1) + (z + 1) * CHUNK_SIZE_WITH_NORMAL_BLENDING;
            verts[vert_counter++] = new Vector3(CHUNK_SIZE_WITH_BORDER, heights[idx], z);
        }

        for (int z = 0; z < CHUNK_SIZE; ++z)
        {
            int left_idx = (CHUNK_SIZE_WITH_BORDER - 1) + (z * CHUNK_SIZE_WITH_BORDER);
            int next_left_idx = (CHUNK_SIZE_WITH_BORDER - 1) + ((z + 1) * CHUNK_SIZE_WITH_BORDER);
            MakeFace(left_idx, right_column_vert_counter, right_column_vert_counter + 1, next_left_idx, tris, ref tri_counter);
            ++right_column_vert_counter;
        }

        return right_column_vert_counter_start;
    }

    private static void DoCorners(NativeArray<Vector3> verts, ref int vert_counter, NativeArray<int> tris, ref int tri_counter, NativeArray<float> heights,
        int top_row, int bottom_row, int left_column, int right_column)
    {
        int top_left = vert_counter++;
        verts[top_left] = new Vector3(-1, heights[0], -1);
        MakeFace(top_left, top_row, 0, left_column, tris, ref tri_counter);

        int top_right = vert_counter++;
        verts[top_right] = new Vector3(CHUNK_SIZE_WITH_BORDER, heights[CHUNK_SIZE_WITH_NORMAL_BLENDING - 1], -1);
        MakeFace(top_row + (CHUNK_SIZE_WITH_BORDER - 1), top_right, right_column, CHUNK_SIZE_WITH_BORDER - 1, tris, ref tri_counter);

        int bottom_right = vert_counter++;
        verts[bottom_right] = new Vector3(CHUNK_SIZE_WITH_BORDER, heights[heights.Length - 1], CHUNK_SIZE_WITH_BORDER);
        MakeFace(CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER - 1, right_column + (CHUNK_SIZE_WITH_BORDER - 1), bottom_right, bottom_row + (CHUNK_SIZE_WITH_BORDER - 1), tris, ref tri_counter);

        int bottom_left = vert_counter++;
        verts[bottom_left] = new Vector3(-1, heights[heights.Length - CHUNK_SIZE_WITH_NORMAL_BLENDING], CHUNK_SIZE_WITH_BORDER);
        MakeFace(left_column + (CHUNK_SIZE_WITH_BORDER - 1), (CHUNK_SIZE_WITH_BORDER - 1) * CHUNK_SIZE_WITH_BORDER, bottom_row, bottom_left, tris, ref tri_counter);
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
