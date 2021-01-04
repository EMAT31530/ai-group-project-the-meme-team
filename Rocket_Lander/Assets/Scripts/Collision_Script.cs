using System;
using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

public class Collision_Script : MonoBehaviour
{

    [SerializeField] private float velocityThresh = 5f;
    [SerializeField] private bool autoQuit = true;
    public GameObject childObject;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    /*void Update()
    {
        Debug.Log(gameObject.GetComponent<Rigidbody>().velocity);
    }*/

    //When the Primitive collides with the walls, it will reverse direction
    private void OnTriggerEnter(Collider other)
    {
        //if (other.name == "Ground")
        if (gameObject.GetComponent<Rigidbody>().velocity.magnitude > velocityThresh)
        {
            gameObject.GetComponent<Renderer>().enabled = false;
            childObject.GetComponent<Renderer>().enabled = false;
            gameObject.GetComponent<Rigidbody>().isKinematic = true;

            if (autoQuit)
                StartCoroutine(Quit());

        }
    }

    // Quit the Unity Editor Play Mode
    IEnumerator Quit()
    {
        yield return new WaitForSeconds(2.5f);
        EditorApplication.isPlaying = false;
    }

}
