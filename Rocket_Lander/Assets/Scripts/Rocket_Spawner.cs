using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Rocket_Spawner : MonoBehaviour
{

    [SerializeField] private float xrange = 50f;
    [SerializeField] private float zrange = 50f;
    [SerializeField] private float ylow = 10f;
    [SerializeField] private float yhigh = 40f;

    [SerializeField] private float tiltAngle = 45f;
    // Assign initial velocity?

    public GameObject go;

    // Start is called before the first frame update
    void Start()
    {

        go.transform.position = new Vector3(Random.Range(-xrange / 2, xrange / 2), Random.Range(ylow, yhigh), Random.Range(-zrange / 2, zrange / 2));
        go.transform.rotation = Quaternion.Euler(Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle));

    }

    // Highlights rocket spawn volume (in blue) when selected
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0, 0, 1, 0.5f);
        Gizmos.DrawCube(new Vector3(0f, (ylow + yhigh) / 2, 0f), new Vector3(xrange, (yhigh - ylow) / 2, zrange));
    }
}