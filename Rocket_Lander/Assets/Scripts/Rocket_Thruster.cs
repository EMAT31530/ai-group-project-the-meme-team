using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class Rocket_Thruster : MonoBehaviour
{

    private Rigidbody rb;
    private Rigidbody rb_thruster;
    public GameObject thruster;
    public ParticleSystem fireParticleSystem;
    private ParticleSystem.EmissionModule em;

    [SerializeField] private float thrustForce = 25f;
    [SerializeField] private float fuel = 1000f;

    [SerializeField] private bool fuelEnabled = false;
    [SerializeField] private bool particlesEnabled = false;

    private Vector3 position_offset; // Store position offset for thruster
    [SerializeField] private bool usingController = true; // True for controller / False for keyboard

    // Start is called before the first frame update
    void Start()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        rb_thruster = thruster.GetComponent<Rigidbody>();
        position_offset = rb_thruster.position - rb.position;
        fuel = 1000f;
        em = fireParticleSystem.emission;
        em.enabled = false;
    }

    // Update is called once per frame
    void FixedUpdate()
    {

        Vector2 tvc_input;
        float throttle;

        // Collect user inputs - continuous with controller, else emulated with keyboard presses
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

        // Rotate the thruster to emulate TVC
        rb_thruster.rotation =
            Quaternion.Euler(Mathf.Atan(tvc_input.x) * Mathf.Rad2Deg, 0, Mathf.Atan(tvc_input.y) * Mathf.Rad2Deg) *
            rb.rotation;
        
        // Apply force at thruster location (hence introducing instability from remote force / moment)
        Vector3 force = transform.TransformVector(rb_thruster.transform.up);
        force = force * throttle * thrustForce;

        // Conduct appropriate thrust logic (e.g. with or without particles and fuel)
        if (fuelEnabled)
        {
            if (fuel > 0 && throttle > 0)
            {
                if(particlesEnabled)
                    em.enabled = true;
                rb.AddForceAtPosition(force, rb_thruster.position);
                fuel -= Vector3.Magnitude(force) / 10;
            }
            else if(particlesEnabled)
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
}
