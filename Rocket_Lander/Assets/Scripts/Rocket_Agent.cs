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

    private bool collisionFlag = false;
    [SerializeField] private float velocityThresh = 5f;

    [SerializeField] private bool randomiseDestination = true;

    // Start is called before the first frame update
    void Start()
    {
        rb = gameObject.GetComponent<Rigidbody>();
        rb_thruster = thruster.GetComponent<Rigidbody>();
        position_offset = rb_thruster.position - rb.position;
        fuel = initial_fuel;
        em = fireParticleSystem.emission;
        em.enabled = false;

        this.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), Random.Range(ylow, yhigh), Random.Range(-zrange / 2, zrange / 2));
        this.transform.localRotation = Quaternion.Euler(Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle));

        if (randomiseDestination)
            destination.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), 1f, Random.Range(-zrange / 2, zrange / 2));
        else
            destination.transform.localPosition = new Vector3(0f, 1f, 0f);
    }

    // Code executed at the beginning of each training episode
    public override void OnEpisodeBegin()
    {

        // Reset position and velocity of agent
        this.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), Random.Range(ylow, yhigh), Random.Range(-zrange / 2, zrange / 2));
        this.transform.localRotation = Quaternion.Euler(Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle), Random.Range(-tiltAngle, tiltAngle));
        this.rb.angularVelocity = Vector3.zero;
        this.rb.velocity = Vector3.zero;
        rb_thruster.position = position_offset + rb.position;

        // Replenish fuel reserves
        fuel = initial_fuel;

        // Reset collision flag
        collisionFlag = false;

        // Randomise destination location
        if (randomiseDestination)
            destination.transform.localPosition = new Vector3(Random.Range(-xrange / 2, xrange / 2), 1f, Random.Range(-zrange / 2, zrange / 2));
        else
            destination.transform.localPosition = new Vector3(0f, 1f, 0f);

    }


    // Code executed each simulation step after receiving actions (heuristic / NN)
    // Replaces the Physics Code, allowing for expedited training
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {

        // Map the Throttle to the expected domain (from [-1,1] to [0,1])
        float mappedThrottle = (actionBuffers.ContinuousActions[2] + 1f) / 2f;

        // Collects vector of actions and stores in actionVector
        Vector3 actionVector = new Vector3(actionBuffers.ContinuousActions[0], actionBuffers.ContinuousActions[1],
            mappedThrottle);

        // Debug Statement Used to Check Mapping of Action Space
        // Debug.Log(actionVector);

        // Calculate and apply thrust using actionVector
        applyThrust(actionVector);

        // Distance to the destination
        float destinationDistance = Vector3.Distance(this.transform.localPosition, destination.transform.localPosition);

        // Rewards
        // If collided (with ground)
        // Mesh collision not perfect => OR low height
        if (collisionFlag || rb.transform.localPosition.y <= 1.2f)
        {
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
            }

            // End episode, and begin next
            EndEpisode();
        }

        // Penalise running out of fuel entirely, and early episode end
        else if (fuelEnabled && fuel <= 0)
        {
            //AddReward(-8.0f);
            AddReward(-1.0f);
            EndEpisode();
        }

        else if (rb.transform.localPosition.y >= 25f)
        {
            AddReward(-5.0f);
            EndEpisode();
        }

        // Award vertical falling (due to stability) - potentially remove?
        /*if (Vector3.Dot(this.transform.up, Vector3.up) > 0.9)
        {
            AddReward(0.01f);
        }*/

        // Penalise use of fuel
        /*if (actionBuffers.ContinuousActions[2] > 0)
        {
            AddReward(-0.01f);
        }*/

        // Add Penalty To Bias Towards Target
        // 2D Position Bias
        //float distance2D = Vector2.Distance(new Vector2(this.transform.localPosition.x, this.transform.localPosition.z), new Vector2(destination.transform.localPosition.x, destination.transform.localPosition.z));
        //float destinationPenalty = (Mathf.Abs((float)Math.Tanh(distance2D / 5.0f)));
        
        // Mapped Penalty with Tanh function
        //float destinationPenalty = (Mathf.Abs((float)Math.Tanh(destinationDistance / 5.0f)));

        // Linear Distance Penalty
        float destinationPenalty = (destinationDistance / 10f);
        AddReward(2f*destinationPenalty);

    }

    // Code for agent to collect observations (sensor readings) from surroundings
    public override void CollectObservations(VectorSensor sensor)
    {

        // Agent's motion properties
        sensor.AddObservation(rb.transform.localPosition);
        sensor.AddObservation(rb.transform.localRotation);
        sensor.AddObservation(rb.velocity);
        sensor.AddObservation(rb.angularVelocity);

        // Fuel levels
        sensor.AddObservation(fuel);

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

        return new Vector3(tvc_input.x, tvc_input.y, throttle);
    }

    // Function to calculate and apply thrust from input vector
    void applyThrust(Vector3 inputVector3)
    {

        // Extract inputs from the input vector
        Vector2 tvc_input = new Vector2(inputVector3.x, inputVector3.y);
        float throttle = inputVector3.z;

        // Rotate the thruster to emulate thrust vectoring
        rb_thruster.transform.localRotation =
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
                rb.AddForceAtPosition(force, rb.transform.localPosition - Vector3.ClampMagnitude(rb.transform.up, 0.1f));
                fuel -= Vector3.Magnitude(force) / 10;
            }
            else if (particlesEnabled)
                em.enabled = false;
        }
        else if (throttle > 0)
        {
            rb.AddForceAtPosition(force, rb.transform.localPosition - Vector3.ClampMagnitude(rb.transform.up, 0.1f));
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
            //Debug.Log("Collision with Ground!!!");
            collisionFlag = true;
        }
    }

    // Gizmo visualisation on selection (for spawn volume)
    void OnDrawGizmosSelected()
    {
        Gizmos.color = new Color(0, 0, 1, 0.5f);
        Gizmos.DrawCube(transform.TransformPoint(new Vector3(0f, (ylow + yhigh) / 2, 0f)), new Vector3(xrange, (yhigh - ylow) / 2, zrange));
    }

}