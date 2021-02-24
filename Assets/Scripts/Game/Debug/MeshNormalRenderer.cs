
using UnityEngine;
using UnityEngine.Assertions;

public class MeshNormalRenderer : MonoBehaviour
{
    public bool m_draw = false;

    public void Update()
    {
        if (!m_draw) return;

        MeshFilter filter = gameObject.GetComponent<MeshFilter>();
        Assert.IsNotNull(filter);

        for (int i = 0; i < filter.mesh.vertexCount; ++i)
        {
            Vector3 vtx_world = gameObject.transform.position + filter.mesh.vertices[i];
            Debug.DrawLine(vtx_world, vtx_world + (filter.mesh.normals[i] * 2.0f));
        }
    }
}
