//using System.Collections;
//using System.Collections.Generic;
using System;
using UnityEngine;
using PUtil.Plot;

public class NNTest : MonoBehaviour
{
    public NeuralNet network;
    public int[] layerlengths = new int[3];
    public bool learn = true;
    public double weightRate = 0.1;
    public double biasRate = 0.1;
    //public Vector<double> input;
    public int epoch = 0;

    // Start is called before the first frame update
    void Start()
    {
        //creat function arrays
        int n = layerlengths.Length;
        Func<double, double>[] activations = new Func<double, double>[n];
        Func<double, double>[] activationDerivatives = new Func<double, double>[n];

        //input the wanted functions
        activations[0] = NeuralNetworkUtil.Identity;
        activationDerivatives[0] = NeuralNetworkUtil.IdentityDerivative;
        for (int i=1; i<n; i++)
        {
            activations[i] = NeuralNetworkUtil.RectifiedLinear;
            activationDerivatives[i] = NeuralNetworkUtil.RectifiedLinearDerivative;
        }

        //create the network
        network = new NeuralNet(layerlengths, activations, activationDerivatives);

        //initialise network with random weights
        network.Initialise();

        print("the weights are");
        for (int i = 0; i < network.weights.Length; i++)
        {
            print("weight at " + i);
            print(network.weights[i].ToString());
        }
    }

    //we're approximating x^2
    public double Error(Vector<double> x, Vector<double> y)//x is input, y is output
    {
        Vector<double> v = new Vector<double>(x.rows);
        double sum = 0;
        v[0] = 0.5 * (x[0] - y[0]) * (x[0] - y[0]);
        sum += v[0];
        for(int i=1; i<v.rows; i++)
        {
            v[i] = 0.5 * (x[i]*x[i] - y[i]) * (x[i]*x[i] - y[i]);
            sum += v[i];
        }
        return sum;

    }

    //derivative for the x^2 error
    public Vector<double> ErrorDerivative(Vector<double> x, Vector<double> y)
    {
        return y-Vector<double>.EWiseProduct(x,x);
    }



    // Update is called once per frame
    void Update()
    {
        if (!learn) return;

        //double[] inputs = PlotUtil.Linspace(0, Math.PI * 2.0, 100);
        double[] inputs = new double[100];
        for(int i=0; i<100; i++)
        {
            if (i % 2 == 0) inputs[i] = 2.0;
            else inputs[i] = 1.0;
        }


        NeuralNetworkUtil.RandomiseArray(inputs); //randomised the inputs
        Vector<double>[] inputVector = new Vector<double>[inputs.Length];
        for (int i = 0; i < inputs.Length; i++)
        {
            inputVector[i] = new Vector<double>(1, new double[] { inputs[i] });
            print("input is  = ");
            print(inputVector[i].ToString());
            network.layers[0] = inputVector[i]; //the input vector

            Vector<double> output = network.ForwardEvaluation();
            print("output is  = ");
            print(output.ToString());
            double err = Error(inputVector[i], output);
            print("the error is = " + err);

            network.Backpropagate(ErrorDerivative, weightRate, biasRate);
            epoch += 1;

            print("After learning at epoch " + epoch + " the weights are");
            for (int j = 0; j < network.weights.Length; j++)
            {
                print("weight at " + j);
                print(network.weights[j].ToString());
            }

            print("________________________________");


        }
    }
}
