using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// Script to spawn in Rocket and Obstacles
public class Obstacle_Spawner : MonoBehaviour
{

    public GameObject obstacle;
    public GameObject rocket;

    [SerializeField] private int obstacle_count = 10;
    [SerializeField] private float xrange = 50f;
    [SerializeField] private float zrange = 50f;
    [SerializeField] private float ylow = 10f;
    [SerializeField] private float yhigh = 40f;
    [SerializeField] private float CLEARANCE_FACTOR = 1.2f;

    // Start is called before the first frame update
    void Start()
    {
        // Get bounding box for rocket - for subsequent distance calculation
        float rocketExtent = rocket.GetComponent<Renderer>().bounds.extents.magnitude;

        int obstacleCounter = 0;

        // Iterate while less obstacles created than requested
        while (obstacleCounter < obstacle_count)
        {
            // Spawn a new obstacle game object in volume defined by xrange, zrange (about the world origin) and from ylow to yhigh in height
            GameObject _obstacle = Instantiate(obstacle, new Vector3(Random.Range(-xrange/2, xrange/2), Random.Range(ylow, yhigh), Random.Range(-zrange/2, zrange/2)), Quaternion.identity);

            // Randomise obstacle scale
            float sf = Random.Range(0.5f, 2f);
            _obstacle.transform.localScale = new Vector3(sf, sf, sf);

            // Get bounding box dimensions for the meteor
            float obstacleExtent = _obstacle.GetComponent<Renderer>().bounds.extents.magnitude;
            
            // Check if distance between rocket and current obstacle is sufficient (so not colliding at start)
            if (Vector3.Distance(rocket.transform.position, _obstacle.transform.position) <=
                (obstacleExtent + rocketExtent * CLEARANCE_FACTOR))
            {
                //Destroy the object if too close
                Destroy(_obstacle);
                continue;
            }

            // Increment iteration counter
            obstacleCounter++;
        }
    }

    // Highlights obstacle spawn volume (in red) when selected
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(1, 0, 0, 0.5f);
        Gizmos.DrawCube(new Vector3(0f, ylow,0f), new Vector3(xrange, (yhigh-ylow)/2, zrange));
    }

}
