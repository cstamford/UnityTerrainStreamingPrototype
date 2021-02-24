using UnityEngine;

public class World : MonoBehaviour
{
    private WorldGen m_world_gen;

    public WorldGen gen => m_world_gen;

    public void Awake()
    {
        m_world_gen = new WorldGen("ceruleanskies");
    }
}
