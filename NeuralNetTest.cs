using System.Collections;
//using System.Collections.Generic;
using System;
using UnityEngine;
using PUtil.Plot;

public class NeuralNetTest : MonoBehaviour
{
    public NeuralNetwork<double> neuralNetwork;
    public int[] layerLengths = new int[3]; //this includes the input output lengths
    public Func<double, double>[] functions;
    public Func<double, double>[] derivativeFunctions;

    public bool learn = true;
    public double[] inputs;

    public double weightRate;
    public double biasRate;

    int epoch;


    // Start is called before the first frame update
    void Start()
    {
        layerLengths[0] = 1;
        layerLengths[layerLengths.Length - 1] = 1;
        functions = new Func<double, double>[layerLengths.Length-1];
        derivativeFunctions = new Func<double, double>[layerLengths.Length-1];
        for(int i=0; i<layerLengths.Length-1; i++)
        {
            if(i==layerLengths.Length-2)
            {
                functions[i] = NeuralNetworkUtil.Identity;
                derivativeFunctions[i] = NeuralNetworkUtil.IdentityDerivative;
            }
            functions[i] = NeuralNetworkUtil.Tanh;
            derivativeFunctions[i] = NeuralNetworkUtil.TanhDerivative;
        }

        neuralNetwork = new NeuralNetwork<double>(layerLengths, functions, derivativeFunctions);
        neuralNetwork.InitialiseNeuralNetwork();
        epoch = 0;
    }

    public static double Error(double x, double y) //x is output, y is target
    {
        return 0.5 * Math.Pow(y - x, 2);
    }
    public static double ErrorDerivative(double x, double y)
    {
        return -y + x;
    }
    public static double SinError(double input, double output)
    {
        return Error(output, Math.Sin(input));
    }

    public static double Cost(double x, double y) //x is input, y is target
    {
        //for neural network x would be the output of the neural network, y would be the function of the input parameters that needs to be approximated
        return 0.5 * Math.Pow(y - x, 2);
    }

    public static double SinCost(double x, double y) //here y is the input for the sin function
    {
        return Cost(x, Math.Sin(y));
    }

    public static double SinVectorCost(Vector<double> x, Vector<double> y)
    {
        double sum = Cost(x[0], Math.Sin(y[0]));
        for(int i=1; i<x.rows; i++)
        {
            sum += Cost(x[i], Math.Sin(y[i]));
        }
        return sum;
    }
    public static Vector<double> SinVectorCostDerivative(Vector<double> x, Vector<double> y)
    {
        Vector<double> c = new Vector<double>(x.rows);
        for(int i=0; i<c.rows; i++)
        {
            c[i] = SinCost(x[i], y[i]);
        }
        return c;
    }


    // Update is called once per frame
    void Update()
    {
        if (!learn) return;
        //Generate inputs
        inputs = PlotUtil.Linspace(0, Math.PI * 2.0, 100);
        NeuralNetworkUtil.RandomiseArray(inputs); //randomised the inputs
        Vector<double>[] inputVector = new Vector<double>[inputs.Length];
        for(int i=0; i<inputs.Length; i++)
        {
            inputVector[i] = new Vector<double>(1, new double[] {inputs[i]});
        }
        double totalError = 0.0;
        Vector<double> costVector = new Vector<double>(inputs.Length);
        for(int i=0; i<inputs.Length; i++)
        {
            Vector<double> v = neuralNetwork.Evaluate(inputVector[i]);
            costVector[i] = SinVectorCost(v, inputVector[i]);
            totalError += SinVectorCost(v, inputVector[i]);
        }
        
        //double error = TotalSinError(v);
        print("total error = " + totalError);
        print("vector errors = " + costVector.ToString());

        neuralNetwork.Backpropagate(inputVector, SinVectorCostDerivative, weightRate, biasRate);
        print("weight 1 = " + neuralNetwork.weights[0].ToString());
        print("weight 2 = " + neuralNetwork.weights[1].ToString());
        epoch++;
        print("epoch = " + epoch);

        print("-------------------------------");
        for(int i=0; i<neuralNetwork.layers.Length; i++)
        {
            print("layer " + i + " = " + neuralNetwork.layers[i].ToString());
        }
        print("next");
    }
}
