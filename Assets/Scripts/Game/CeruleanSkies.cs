using UnityEngine;
using UnityEngine.Assertions;

public class CeruleanSkies : MonoBehaviour
{
    public void Awake()
    {
        // TODO: Should happen when entering from UI -> Game
        GameObject world_object = new GameObject("World", typeof(World));
        World world = world_object.GetComponent<World>();
        world.Init("ceruleanskies", GameObject.Find("MainCamera"));
    }
}
