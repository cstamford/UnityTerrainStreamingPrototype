
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
    public const int CHUNK_SIZE_WITH_BORDER = CHUNK_SIZE + 1;

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
            heights[id] = 2.0f *
                Mathf.Max(
                    perlin.GetHeightAt(coords[id].x / 4.0f, coords[id].y / 4.0f),
                    perlin.GetHeightAt(coords[id].x / 8.0f, coords[id].y / 8.0f)) +
                Mathf.Min(
                    perlin.GetHeightAt(coords[id].x / 16.0f, coords[id].y / 16.0f),
                    perlin.GetHeightAt(coords[id].x / 32.0f, coords[id].y / 32.0f));
        }
    }

    public static JobHandle ScheduleChunkGeneration(int chunk_x, int chunk_z, Lod lod, PerlinGenerator perlin, out NativeArray<float> heights)
    {
        int lod_step = (int)lod;

        NativeArray<Vector2> coords = new NativeArray<Vector2>(CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER / lod_step, Allocator.TempJob);

        int i = 0;
        for (int z = chunk_z; z < chunk_z + CHUNK_SIZE_WITH_BORDER; z += lod_step)
        {
            for (int x = chunk_x; x < chunk_x + CHUNK_SIZE_WITH_BORDER; x += lod_step)
            {
                coords[i++] = new Vector2(x, z);
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
        int lod_step = (CHUNK_SIZE_WITH_BORDER * CHUNK_SIZE_WITH_BORDER) / heights.Length;
        int stride = CHUNK_SIZE_WITH_BORDER / lod_step;

        // one vert per height point
        Vector3[] verts = new Vector3[heights.Length];
        Vector2[] uvs = new Vector2[heights.Length];

        for (int z = 0; z < stride; ++z)
        {
            for (int x = 0; x < stride; ++x)
            {
                int idx = x + z * stride;
                verts[idx] = new Vector3(x * lod_step, heights[idx], z * lod_step);
                uvs[idx] = new Vector2(x / (float)stride, z / (float)stride);
            }
        }

        int grid_stride = CHUNK_SIZE / lod_step;

        // 2 triangles per grid + 3 verts per triangle
        int[] tris = new int[heights.Length * 2 * 3];

        for (int z = 0; z < grid_stride; ++z)
        {
            for (int x = 0; x < grid_stride; ++x)
            {
                int vert_idx = x + z * stride;
                int tri_idx = vert_idx * 6;

                int t0 = vert_idx;
                int t1 = vert_idx + 1;
                int t2 = vert_idx + stride;
                int t3 = vert_idx + 1;
                int t4 = vert_idx + 1 + stride;
                int t5 = vert_idx + stride;

                Assert.IsTrue(t0 >= 0 && t0 < verts.Length);
                Assert.IsTrue(t1 >= 0 && t1 < verts.Length);
                Assert.IsTrue(t2 >= 0 && t2 < verts.Length);
                Assert.IsTrue(t3 >= 0 && t3 < verts.Length);
                Assert.IsTrue(t4 >= 0 && t4 < verts.Length);
                Assert.IsTrue(t5 >= 0 && t5 < verts.Length);

                tris[tri_idx + 0] = t0;
                tris[tri_idx + 1] = t2;
                tris[tri_idx + 2] = t1;
                tris[tri_idx + 3] = t3;
                tris[tri_idx + 4] = t5;
                tris[tri_idx + 5] = t4;
            }
        }

        Mesh mesh = new Mesh()
        {
            vertices = verts,
            uv = uvs,
            triangles = tris
        };

        mesh.RecalculateNormals();
        return mesh;
    }
}
