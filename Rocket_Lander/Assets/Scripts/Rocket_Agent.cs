using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using UnityEngine.Apple;
using UnityEngine.InputSystem;
using Random = UnityEngine.Random;

public class Rocket_Agent : Agent
{

    // Parameter Initialisations
    private Rigidbody rb;
    private Rigidbody rb_thruster;
    public GameObject thruster;
    public GameObject destination;
    public ParticleSystem fireParticleSystem;
    private ParticleSystem.EmissionModule em;
    private Vector3 position_offset;

    // TVC Parameters
    [SerializeField] private float thrustForce = 25000f;
    [SerializeField] private bool particlesEnabled = true;
    [SerializeField] private bool usingController = true;

    // Spawner Parameters
    [SerializeField] private float xrange = 50f;
    [SerializeField] private float zrange = 50f;
    [SerializeField] private float ylow = 10f;
    [SerializeField] private float yhigh = 40f;
    [SerializeField] private float tiltAngle = 45f;

    private bool collisionFlag = false;
    [SerializeField] private float velocityThresh = 5f;
    [SerializeField] private bool randomiseDestination = true;
    [SerializeField] private bool thrusterTorque = true;

    // Start is called before the first frame update
    void Start()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        rb_thruster = thruster.GetComponent<Rigidbody>();
        position_offset = rb_thruster.position - rb.position;
        em = fireParticleSystem.emission;
        em.enabled = false;

        this.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), Random.Range(ylow, yhigh), Random.Range(-zrange / 2, zrange / 2));
        this.transform.localRotation = Quaternion.Euler(Random.Range(-tiltAngle, tiltAngle), 0, Random.Range(-tiltAngle, tiltAngle));

        if (randomiseDestination)
            destination.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), 1f, Random.Range(-zrange / 2, zrange / 2));
        else
            destination.transform.localPosition = new Vector3(0f, 1f, 0f);

        rb_thruster.position = position_offset + rb.position;
    }

    // Code executed at the beginning of each training episode
    public override void OnEpisodeBegin()
    {

        // Reset position and velocity of agent
        this.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), Random.Range(ylow, yhigh), Random.Range(-zrange / 2, zrange / 2));
        this.transform.localRotation = Quaternion.Euler(Random.Range(-tiltAngle, tiltAngle), 0, Random.Range(-tiltAngle, tiltAngle));
        this.rb.angularVelocity = Vector3.zero;

        // Non-zero initial velocity so as if already falling for a while
        this.rb.velocity = new Vector3(0f,-10f,0f);
        rb_thruster.position = position_offset + rb.position;

        // Reset collision flag
        collisionFlag = false;

        // Randomise destination location
        if (randomiseDestination)
            destination.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), 1f, Random.Range(-zrange / 2, zrange / 2));
        else
            destination.transform.localPosition = new Vector3(0f, 1f, 0f);

        rb_thruster.position = position_offset + rb.position;
    }


    // Code executed each simulation step after receiving actions (heuristic / NN)
    // Replaces the Physics Code, allowing for expedited training
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        // Clamp the Throttle between [-1,1] for stability of continuous sampling
        float clampedThrottle = Mathf.Clamp(actionBuffers.ContinuousActions[2], -1f, 1f);

        // Map the Throttle to the expected domain (from [-1,1] to [0,1])
        float mappedThrottle = (clampedThrottle + 1f) / 2f;

        // Collects vector of actions and stores in actionVector
        Vector3 actionVector = new Vector3(actionBuffers.ContinuousActions[0], actionBuffers.ContinuousActions[1], mappedThrottle);

        // Calculate and apply thrust using actionVector
        applyThrust(actionVector);

        // Distance to the destination
        float destinationDistance = Vector3.Distance(this.transform.localPosition, destination.transform.localPosition);

        // If collided (with ground)
        if (collisionFlag)
        {
            /* For now, simplifying reward function to single term
            // Check and reward low speed and vertical landing conditions
            // Create binary flags using ternary operators
            int speedCriterion = rb.velocity.magnitude < velocityThresh ? 1:0;
            int orientationCriterion = Mathf.Abs(Vector3.Dot(this.transform.up, Vector3.up)) > 0.8 ? 1 : 0;
            int distanceCriterion = destinationDistance < 2 ? 1 : 0;

            // Reward based on adherance to speed, distance, and orientation criterion
            if (distanceCriterion == 1)
            {
                AddReward((speedCriterion * 4.0f + orientationCriterion * 2.0f) + 6.0f);
            }
            else
            {
                AddReward(-2f);
            }*/

            // Could add a reward here before ending episode to quantify
            // speed and orientation, as above

            // End episode, and begin next
            EndEpisode();
        }

        // Penalise and end episode if leave bounds of training area
        else if (rb.transform.localPosition.y >= 25f || rb.transform.localPosition.x >= 65f ||
                 rb.transform.localPosition.x <= -65f || rb.transform.localPosition.z >= 65f ||
                 rb.transform.localPosition.z <= -65f)
        {
            AddReward(-5.0f);
            EndEpisode();
        }
        
        // Linear Distance Penalty
        AddReward(-1 * (destinationDistance / 10f));

    }

    // Code for agent to collect observations (sensor readings) from surroundings
    public override void CollectObservations(VectorSensor sensor)
    {

        // Agent's motion properties
        sensor.AddObservation(rb.transform.localPosition);
        sensor.AddObservation(rb.transform.localRotation);
        sensor.AddObservation(rb.velocity);
        sensor.AddObservation(rb.angularVelocity);

        // Destination Location
        sensor.AddObservation(destination.transform.localPosition);

    }

    // Code to define heuristic behaviour => I.e. provide manual control
    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var continuousActionsOut = actionsOut.ContinuousActions;
        Vector3 inputsVector3 = gatherInputs();
        continuousActionsOut[0] = inputsVector3.x;
        continuousActionsOut[1] = inputsVector3.y;
        continuousActionsOut[2] = inputsVector3.z;

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

        // Map the Throttle to consistent domain as ML Agent (from [0,1] to [-1,1])
        throttle = throttle * 2 - 1;

        return new Vector3(tvc_input.x, tvc_input.y, throttle);
    }

    // Function to calculate and apply thrust from input vector
    void applyThrust(Vector3 inputVector3)
    {

        // Extract inputs from the input vector
        Vector2 tvc_input = new Vector2(inputVector3.x, inputVector3.y);
        float throttle = inputVector3.z;

        // Rotate the thruster to emulate thrust vectoring
        rb_thruster.transform.rotation = transform.rotation *
                                         Quaternion.Euler(Mathf.Atan(tvc_input.y) * Mathf.Rad2Deg, 0,
                                             Mathf.Atan(tvc_input.x) * Mathf.Rad2Deg);

        // Calculate thrust force to be applied
        Vector3 force = rb_thruster.transform.up * throttle * thrustForce;

        // Conduct appropriate thrust logic (e.g. with or without particles and fuel)
        if (throttle > 0)
        {

            if (!thrusterTorque)
            {
                // Force w/o turning moment (effectively 3 DOF, translational)
                rb.AddForce(force);
            }
            else
            {
                // Force w/ turning moment (full 6 DOF, translational + rotational)
                rb.AddForceAtPosition(force, transform.TransformPoint(rb.transform.parent.localPosition));
            }

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
            collisionFlag = true;
        }
    }

}