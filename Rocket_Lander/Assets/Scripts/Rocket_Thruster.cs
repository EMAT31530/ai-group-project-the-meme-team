using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;
using Random = UnityEngine.Random;

public class Rocket_Thruster : MonoBehaviour
{

    // Parameter Initialisations
    private Rigidbody rb;
    private Rigidbody rb_thruster;
    public GameObject thruster;
    public ParticleSystem fireParticleSystem;
    private ParticleSystem.EmissionModule em;
    private Vector3 position_offset;
    private float fuel;

    // TVC Parameters
    [SerializeField] private float thrustForce = 25000f;
    [SerializeField] private float initial_fuel = 100000f;
    [SerializeField] private bool fuelEnabled = true;
    [SerializeField] private bool particlesEnabled = true;
    [SerializeField] private bool usingController = true;

    // Spawner Parameters
    [SerializeField] private float xrange = 50f;
    [SerializeField] private float zrange = 50f;
    [SerializeField] private float ylow = 10f;
    [SerializeField] private float yhigh = 40f;
    [SerializeField] private float tiltAngle = 45f;

    // Start is called before the first frame update
    void Start()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        rb_thruster = thruster.GetComponent<Rigidbody>();
        position_offset = rb_thruster.position - rb.position;
        fuel = initial_fuel;
        em = fireParticleSystem.emission;
        em.enabled = false;

        this.transform.position = new Vector3(Random.Range(-xrange / 2, xrange / 2), Random.Range(ylow, yhigh), Random.Range(-zrange / 2, zrange / 2));
        this.transform.rotation = Quaternion.Euler(Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle));
    }

    // Physics code - execute each fixed update (0.02 seconds)
    void FixedUpdate()
    {

        Vector3 inputVector3 = gatherInputs();
        applyThrust(inputVector3);

    }

    // Gizmo visualisation on selection (for spawn volume)
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0, 0, 1, 0.5f);
        Gizmos.DrawCube(new Vector3(0f, (ylow + yhigh) / 2, 0f), new Vector3(xrange, (yhigh - ylow) / 2, zrange));
    }

    // Function to collect inputs from the user / agent
    Vector3 gatherInputs()
    {
        Vector2 tvc_input;
        float throttle;

        if (usingController)
        {
            var gamepad = Gamepad.current;
            tvc_input = gamepad.leftStick.ReadValue();
            throttle = gamepad.leftTrigger.ReadValue();
        }
        else
        {
            var keyboard = Keyboard.current;
            int right = Convert.ToInt16(keyboard.rightArrowKey.IsPressed());
            int left = Convert.ToInt16(keyboard.leftArrowKey.IsPressed());
            int up = Convert.ToInt16(keyboard.upArrowKey.IsPressed());
            int down = Convert.ToInt16(keyboard.downArrowKey.IsPressed());
            tvc_input = new Vector2(right - left, up - down);
            throttle = (float)Convert.ToInt16(keyboard.spaceKey.IsPressed());
        }

        return new Vector3(tvc_input.x, tvc_input.y, throttle);
    }

    // Function to calculate and apply thrust from input vector
    void applyThrust(Vector3 inputVector3)
    {

        // Extract inputs from the input vector
        Vector2 tvc_input = new Vector2(inputVector3.x, inputVector3.y);
        float throttle = inputVector3.z;

        // Rotate the thruster to emulate thrust vectoring
        rb_thruster.rotation =
            Quaternion.Euler(Mathf.Atan(tvc_input.x) * Mathf.Rad2Deg, 0, Mathf.Atan(tvc_input.y) * Mathf.Rad2Deg) *
            rb.rotation;

        // Calculate thrust force to be applied
        Vector3 force = rb_thruster.transform.up * throttle * thrustForce;

        // Conduct appropriate thrust logic (e.g. with or without particles and fuel)
        if (fuelEnabled)
        {
            if (fuel > 0 && throttle > 0)
            {
                if (particlesEnabled)
                    em.enabled = true;

                // Apply force at offset location (hence introducing instability from remote force / moment)
                rb.AddForceAtPosition(force, rb.position - Vector3.ClampMagnitude(rb.transform.up, 0.1f));
                fuel -= Vector3.Magnitude(force) / 10;
            }
            else if (particlesEnabled)
                em.enabled = false;
        }
        else if (throttle > 0)
        {
            rb.AddForceAtPosition(force, rb_thruster.position);
            if (particlesEnabled)
                em.enabled = true;
        }
        else if (particlesEnabled)
            em.enabled = false;

        rb_thruster.position = position_offset + rb.position;

    }

    // Collision Event Handling
    void OnCollisionEnter(Collision other)
    {
        if (other.gameObject.name == "Ground")
        //if (gameObject.GetComponent<Rigidbody>().velocity.magnitude > velocityThresh)
        {
            Debug.Log("Collision with Ground!!!");
        }
    }
}
