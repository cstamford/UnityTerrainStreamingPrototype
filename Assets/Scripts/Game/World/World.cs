using UnityEngine;

public class World : MonoBehaviour
{
    private WorldGen m_world_gen;
    private WorldStreaming m_world_streaming;
    private GameObject m_camera;

    public void Init(string seed, GameObject camera)
    {
        m_world_gen = new WorldGen(seed);
        m_world_streaming = new WorldStreaming(m_world_gen);
        m_camera = camera;
    }

    public void Update()
    {
        m_world_streaming.Update(m_camera.transform.position);
    }
}
