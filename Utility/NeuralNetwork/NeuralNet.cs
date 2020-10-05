using System;
using PUtil.Operators;

public class NeuralNet
{
    public Vector<double>[] layers; //each layer including the input and output vectors
    public Matrix<double>[] weights; //weights between each vector
    public Vector<double>[] biases; //biases between each vector

    public Func<Vector<double>, Vector<double>>[] afuncs; //activation functions for each layer - Excluding the input
    public Func<Vector<double>, Vector<double>>[] dfuncs; //derivative of activation functions - Excluding the input

    public NeuralNet(int[] lengths, Func<double, double>[] _afuncs, Func<double, double>[] _dfuncs)
    {
        if (lengths.Length < 2) throw new Exception("At least 2 layers are required for input and output");
        int n = lengths.Length;

        layers = new Vector<double>[n]; //we want N many layers, this includes in/output
        weights = new Matrix<double>[n - 1]; //we want N-1 many weights as there's N-1 transitions
        biases = new Vector<double>[n - 1]; //Again because we have N-1 transitions

        afuncs = new Func<Vector<double>, Vector<double>>[n]; //because we require activation function for input - might be changed
        dfuncs = new Func<Vector<double>, Vector<double>>[n]; //same as above

        for(int i=0; i<n; i++)
        {
            layers[i] = new Vector<double>(lengths[i]);
            afuncs[i] = NeuralNetworkUtil.CreateVectorFunction(_afuncs[i]); //this just translates the double function to do this activation for each of the elements of the vector
            dfuncs[i] = NeuralNetworkUtil.CreateVectorFunction(_dfuncs[i]); //the indexies as follows W(i) * A(i) + B(i) = V(i+1) => F[i+1](V(i+1)) = A(i+1)

            if (i < 1) continue;

            weights[i-1] = new Matrix<double>(lengths[i], lengths[i-1]); //compatible to multiple as W(i) * A(i) + B(i) = V(i+1), where W is weights, V are the un-activated layer, B is bias, A is activated layer
            biases[i-1] = new Vector<double>(lengths[i]); //same as above
        }
    }

    public void Initialise(int seed = 0)
    {
        Random rand;
        if (seed == 0) rand = new Random();
        else rand = new Random(seed);

        for (int i = 0; i < weights.Length; i++)
        {
            for (int j = 0; j < weights[i].rows; j++)
            {
                for (int k = 0; k < weights[i].columns; k++)
                {
                    weights[i][j, k] = rand.NextDouble() / 10.0;
                }
            }
        }
    }

    public Vector<double> EvaluateLayer(int i)
    {
        //W(i) * A(i) + B(i) = V(i+1) => A(i+1) = F[i](V(i+1))
        //where W is weights, V are the un-activated layer, B is bias, A is activated layer
        return weights[i] * afuncs[i](layers[i]) + biases[i]; //this was written in the initialise bit
    }

    public Vector<double> ForwardEvaluation() //here we assumed that layers[0] is already is set
    {
        int n = layers.Length-1;
        for (int i=0; i<n; i++)
        {
            if(i == 0) { layers[i + 1] = weights[i] * afuncs[i](layers[i]) + biases[i]; continue; } //because for now we assumed that the first layer "the input", does require activation

            layers[i + 1] = EvaluateLayer(i); //just the usual forward evaluation, but we store in layers the un-activated vectors
        }
        return afuncs[n](layers[n]); //return the activated output layer
    }

    public Vector<double> SigmaEnd(Func<Vector<double>, Vector<double>, Vector<double>> dCost) //DCost is the derivative of the cost function
    {
        //since we assumed that the input/first layer is already activated
        //Cost(input, output) = Error
        int n = layers.Length - 1;
        Vector<double> costs = dCost(afuncs[0](layers[0]), afuncs[n](layers[n])); //this is the partial derivatives of the cost function with respect to the outputs
        return new Vector<double>(Matrix<double>.EWiseProduct(costs, dfuncs[n](layers[n]))); //because the last layers is the non-activated output function
    }

    public Vector<double> SigmaInner(int l, Vector<double> sigma) //l is the layer
    {
        if (l >= layers.Length - 1) throw new Exception("this is the last layer, can't apply method");
        return new Vector<double>(Matrix<double>.EWiseProduct(Matrix<double>.Transpose(weights[l+1]) * sigma, dfuncs[l+1](layers[l+1])));
    }

    public void UpdateWeightBiases(Vector<double>[] sigmas, double weightRate, double biasRate)
    {
        int n = weights.Length;
        for(int i=0; i<n; i++)
        {
            biases[i] -= sigmas[i] * biasRate;
            weights[i] -= (sigmas[i] * Matrix<double>.Transpose(afuncs[i](layers[i]))) * weightRate;
        }
    }

    public void Backpropagate(Func<Vector<double>, Vector<double>, Vector<double>> dCost, double weightRate, double biasRate) //we need the cost function to backpropagate with respect to
    {
        ForwardEvaluation();
        int n = weights.Length;
        Vector<double>[] sigmas = new Vector<double>[n];
        sigmas[n - 1] = SigmaEnd(dCost);
        for(int i=n-2; i>=0; i--) //we go backwards
        {
            sigmas[i] = SigmaInner(i, sigmas[i + 1]);
        }
        UpdateWeightBiases(sigmas, weightRate, biasRate); //finished backpropagation
    }

}
