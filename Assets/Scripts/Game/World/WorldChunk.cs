using System;
using Unity.Collections;
using Unity.Jobs;
using UnityEngine;

public class WorldChunk
{
    public enum Lod
    {
        FullQuality = 1,
        HalfQuality = 2,
        QuarterQuality = 4
    }

    public const int CHUNK_SIZE = 32;
    private const int CHUNK_SIZE_WITH_BORDER = CHUNK_SIZE + 1;
    private const int CHUNK_SIZE_WITH_NORMAL_BLENDING = CHUNK_SIZE_WITH_BORDER + 2;

    private struct GenerateChunkJob : IJobParallelFor
    {
        [ReadOnly]
        [DeallocateOnJobCompletion]
        public NativeArray<Vector2> coords;

        [ReadOnly]
        public PerlinGenerator perlin;

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
        NativeArray<Vector2> coords = new NativeArray<Vector2>(CHUNK_SIZE_WITH_NORMAL_BLENDING * CHUNK_SIZE_WITH_NORMAL_BLENDING, Allocator.TempJob);

        int i = 0;
        for (int z = chunk_z; z < chunk_z + CHUNK_SIZE_WITH_NORMAL_BLENDING; ++z)
        {
            for (int x = chunk_x; x < chunk_x + CHUNK_SIZE_WITH_NORMAL_BLENDING; ++x)
            {
                coords[i++] = new Vector2(x - 1, z - 1);
            }
        }

        heights = new NativeArray<float>(coords.Length, Allocator.TempJob);

        GenerateChunkJob job = new GenerateChunkJob()
        { 
            coords = coords,
            perlin = perlin,
            heights = heights
        };

        return job.Schedule(coords.Length, 32);
    }

    // TODO: Maybe need to jobify?
    public static Mesh GenerateChunkMesh(NativeArray<float> heights)
    {   
        Vector3[] verts = new Vector3[heights.Length];
        Vector2[] uvs = new Vector2[CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER];
        Vector3[] normals = new Vector3[CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER];
        int[] tris = new int[heights.Length * 2 * 3];

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

        // -- Step 3:
        // Calculate normals to prevent seams between neighbouring chunks.

        for (int i = 0; i < normals.Length; ++i)
        {
            for (int tri = 0; tri < tri_counter; tri += 3)
            {
                int t0 = tris[tri + 0];
                int t1 = tris[tri + 1];
                int t2 = tris[tri + 2];

                if (t0 == i || t1 == i || t2 == i)
                {
                    Vector3 t0v = verts[t0];
                    Vector3 t1v = verts[t1];
                    Vector3 t2v = verts[t2];

                    Vector3 n = Vector3.Cross(t1v - t0v, t2v - t0v);

                    const bool USE_WEIGHTED_NORMALS = true;

                    if (USE_WEIGHTED_NORMALS)
                    {
                        // TODO: I'm not so sure about this code, but it seems to work decently...
                        // See https://stackoverflow.com/questions/45477806/general-method-for-calculating-smooth-vertex-normals-with-100-smoothness

                        float a1 = Vector3.Angle(t1v - t0v, t2v - t0v);
                        float a2 = Vector3.Angle(t2v - t1v, t0v - t1v);
                        float a3 = Vector3.Angle(t0v - t2v, t1v - t2v);

                        normals[i] += n * a1;
                        normals[i] += n * a2;
                        normals[i] += n * a3;
                    }
                    else
                    {
                        normals[i] += n;
                    }
                }
            }

            normals[i].Normalize();
        }


        // -- Step 4:
        // Remove the outside tape, so we're left with the good mesh.

        Mesh mesh = new Mesh();
        mesh.SetVertices(verts, 0, CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER);
        mesh.SetUVs(0, uvs);
        mesh.SetNormals(normals);

        Array.Resize(ref tris, CHUNK_SIZE * CHUNK_SIZE * 2 * 3);
        mesh.SetTriangles(tris, 0);

        return mesh;
    }

    private static int DoTopRow(Vector3[] verts, ref int vert_counter, int[] tris, ref int tri_counter, NativeArray<float> heights)
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

    private static int DoBottomRow(Vector3[] verts, ref int vert_counter, int[] tris, ref int tri_counter, NativeArray<float> heights)
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

    private static int DoLeftColumn(Vector3[] verts, ref int vert_counter, int[] tris, ref int tri_counter, NativeArray<float> heights)
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

    private static int DoRightColumn(Vector3[] verts, ref int vert_counter, int[] tris, ref int tri_counter, NativeArray<float> heights)
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

    private static void DoCorners(Vector3[] verts, ref int vert_counter, int[] tris, ref int tri_counter, NativeArray<float> heights,
        int top_row, int bottom_row, int left_column, int right_column)
    {
        // Top left
        int top_left = vert_counter++;
        verts[top_left] = new Vector3(-1, heights[0], -1);
        MakeFace(top_left, top_row, 0, left_column, tris, ref tri_counter);

        // Top right
        int top_right = vert_counter++;
        verts[top_right] = new Vector3(CHUNK_SIZE_WITH_BORDER, heights[CHUNK_SIZE_WITH_NORMAL_BLENDING - 1], -1);
        MakeFace(top_row + (CHUNK_SIZE_WITH_BORDER - 1), top_right, right_column, CHUNK_SIZE_WITH_BORDER - 1, tris, ref tri_counter);

        // Bottom right
        int bottom_right = vert_counter++;
        verts[bottom_right] = new Vector3(CHUNK_SIZE_WITH_BORDER, heights[heights.Length - 1], CHUNK_SIZE_WITH_BORDER);
        MakeFace(CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER - 1, right_column + (CHUNK_SIZE_WITH_BORDER - 1), bottom_right, bottom_row + (CHUNK_SIZE_WITH_BORDER - 1), tris, ref tri_counter);

        // Bottom left
        int bottom_left = vert_counter++;
        verts[bottom_left] = new Vector3(-1, heights[heights.Length - CHUNK_SIZE_WITH_NORMAL_BLENDING], CHUNK_SIZE_WITH_BORDER);
        MakeFace(left_column + (CHUNK_SIZE_WITH_BORDER - 1), (CHUNK_SIZE_WITH_BORDER - 1) * CHUNK_SIZE_WITH_BORDER, bottom_row, bottom_left, tris, ref tri_counter);
    }

    private static void MakeFace(int v0, int v1, int v2, int v3, int[] tris, ref int tri_counter)
    {
        tris[tri_counter++] = v0;
        tris[tri_counter++] = v3;
        tris[tri_counter++] = v1;
        tris[tri_counter++] = v1;
        tris[tri_counter++] = v3;
        tris[tri_counter++] = v2;
    }
}
