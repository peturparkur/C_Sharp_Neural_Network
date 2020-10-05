using System;
//using System.Numerics;
using PUtil.Operators;

public class NeuralNetwork<T> where T:new()
{
    public int hiddenlayers; //we need atleast 2 layers, one for input, one for output
    public Vector<T>[] layers; //each layer
    //we need to know size of input vector
    //we need size of output vector
    //preferably the size of hidden layers
    //the layer just before the output layer should be the same size as we should just scale it to get output
    //the output function should be the identity function unless specified otherwise
    //represent each weight between layers as matrices
    public Matrix<T>[] weights; //weight matrices
    public Vector<T>[] biases;
    public Func<T, T>[] funcs; //we should have a same amount of weights and functions
    public Func<T, T>[] dfuncs;

    //weights[0] multiplies vector[0]

    public NeuralNetwork(int[] layerLength, Func<T,T>[] f, Func<T,T>[] dF)
    {
        if (f.Length != layerLength.Length - 1) throw new Exception("Functions don't have matching length for neural network");
        //we assume layerLength.length >= 2 if it is 2 we only how input and output
        int n = layerLength.Length;
        layers = new Vector<T>[n];
        weights = new Matrix<T>[n - 1];
        funcs = new Func<T, T>[n - 1];
        dfuncs = new Func<T, T>[n - 1];
        biases = new Vector<T>[n - 1];
        for(int i=0; i<n; i++)
        {
            layers[i] = new Vector<T>(layerLength[i]);

            if(i>=1)
            {
                weights[i - 1] = new Matrix<T>(layers[i].rows, layers[i-1].rows); //create a matrix which matches the dimensions of the vectors to be compatible
                //meaning if layer[i-1] has N rows and layer[i] has M rows we create a MatrixMxN, M rows, N columns.

                funcs[i - 1] = f[i-1]; //the activation function of layer i
                dfuncs[i - 1] = dF[i - 1];
                biases[i - 1] = new Vector<T>(layerLength[i], true);
            }
        }
    }

    public void InitialiseNeuralNetwork(int seed = 0) //use this method to add some small random weights and biases
    {
        Random rand;
        if (seed == 0) rand = new Random();
        else rand = new Random(seed);
        
        for(int i=0; i<weights.Length; i++)
        {
            for(int j=0; j<weights[i].rows; j++)
            {
                for(int k=0; k<weights[i].columns; k++)
                {
                    weights[i][j, k] = Operator<double, T>.convert(rand.NextDouble() / 5);
                }
            }
        }
    }

    public void SetLayer(int i, Vector<T> layer) //might now be necessary
    {
        if (layers[i].rows != layer.rows) throw new System.Exception("dimension doesn't match");
        layers[i] = new Vector<T>(layer);
    }

    public Vector<T> GetLayer(int i) //might now be necessary
    {
        return new Vector<T>(layers[i]);
    }

    public Vector<T> EvaluateLayer(Vector<T> input, int i) //i is the layer number
    {
        if (i >= layers.Length - 1) throw new System.Exception("over the weight layers, i points to after input layer");
        Vector<T> v = new Vector<T>(weights[i] * ActivateVector(layers[i], funcs[i]) + biases[i]); //we need this for the activation function
        /*for(int j=0; j<v.rows; j++) //calling the activation function
        {
            v[j] = funcs[i](v[j]); //applying activation function
        }*/
        return v; //weights * vector + bias
        //return new Vector<T>(weights[i] * layers[i]);
    }

    public Vector<T> Evaluate(Vector<T> input) //we just evaluate each layer then return the last one as the output
    {

        for(int i=0; i<layers.Length-2; i++)
        {
            layers[i + 1] = EvaluateLayer(input, i); //we evaluate each layer up to the end
        }
        return new Vector<T>(layers[layers.Length - 1]); //we only want to return a copy not the reference
    }

    public static Vector<T> ActivateVector(Vector<T> v, Func<T, T> activateF) //apply activation function on the vector
    {
        Vector<T> a = new Vector<T>(v.rows); //create a copy of the vector
        for(int i=0; i<v.rows; i++)
        {
            a[i] = activateF(a[i]); //apply the activation function on the vector
        }
        return a;
    }

    //backpropagation should be
    //vector the vector before the activation function, f(vector) is the activation function applied on the vector
    //d(Error)/d(weight) = d(Error)/d(f(vector)) * d(f(vector))/d(weight) = d(Error)/d(f(vector)) * d(f(vector))/d(vector) * d(vector)/d(weight)
    //=> d(vector)/d(weight) = f(prevVector)
    //=> d(f(vector))/d(vector) = d(func(vector))/d(vector) = d(func)(vector) --> derivative of func at prevVector
    //=> d(Error)/d(f(vector)) --> if Error(x,y) = (1/2) * (x-y)^2 ==> d(Error)/d(Vector) = y-t
    //=> we update weights by -alpha * d(Error)/d(weight)
    //=> usually denoted d(Error)/d(f(vector)) * d(f(vector))/d(vector) = delta(j) for the jth vector
    //==> delta(j) = d(Error)/d(f(vector)) * d(f(vector))/d(vector) if j is output neuron
    //==> delta(j) = sum(weights(j) * delta(j+1)) * d(f(vector))/d(vector) if j is an inner neuron
    //==> Then d(Error)/d(weight(i,j)) = delta(j)*f(vector(i))
    //==> to have explicit derivatives we need to know the exact error function and the exact activation function

    //Function to get delta for the output layers, this is used for backpropagation
    public Vector<T> SigmaEnd(Func<Vector<T>,Vector<T>, Vector<T>> dCost, Func<Vector<T>,Vector<T>> dFunc, Vector<T> output) //dCost = derivative of cost function with respect to an output neuron, dFunc is derivative of activation function with respect to output neuron pre-activation
    {
        int n = output.rows;
        //int n = layers[layers.Length - 1].rows; //length of output neuron
        //Vector<T> delta = new Vector<T>(layers[layers.Length - 1].rows); //length of output vector
        //Vector<T> costVector = layers[0];
        Vector<T> dVector = new Vector<T>(output);

        /*for(int i=0; i<n; i++)
        {
            costVector[i] = dCost(funcs[n-1](dVector[i])); //derivative of cost function with respect to output neuron i after activation
            dVector[i] = dFunc(dVector[i]); //derivative of activation function of output neuron i before activation
        }
        */
        return new Vector<T>(Matrix<T>.EWiseProduct(dCost(dVector, layers[0]), dFunc(dVector))); //delta for the output layer
    }

    public static Vector<T> SigmaInner(Vector<T> vector, Matrix<T> weightNext, Vector<T> sigmaNext, Func<Vector<T>,Vector<T>> dFunc) //dFunc is derivative of activation function
    {
        return new Vector<T>(Matrix<T>.EWiseProduct(Matrix<T>.Transpose(weightNext)*sigmaNext, dFunc(vector)));
    }

    public Vector<T>[] CalculateSigma(Func<Vector<T>, Vector<T>, Vector<T>> dCost) //dCost = derivative of cost function
    {
        int n = layers.Length-1;
        Vector<T>[] sm = new Vector<T>[n];
        sm[n-1] = SigmaEnd(dCost, NeuralNetworkUtil.CreateVectorFunction(dfuncs[n-1]), layers[n]);
        for(int i=n-2; i>=0; i--)
        {
            sm[i] = SigmaInner(layers[i + 1], weights[i + 1], sm[i + 1], NeuralNetworkUtil.CreateVectorFunction(dfuncs[i]));
        }
        //sm is all the delta/sigma for each weight
        return sm;
    }

    public void UpdateWeightsBiases(Vector<T>[] sigmas, T weightRate, T biasRate)
    {
        int n = weights.Length;
        for(int i=n-1; i>=0; i--) //looping backward
        {
            biases[i] -= sigmas[i] * biasRate;
            weights[i] -= (Matrix<T>)sigmas[i] * Matrix<T>.Transpose(ActivateVector(layers[i], funcs[i])) * weightRate;

            /*biases[i] -= sigmas[i] * biasRate;
            for(int j=0; j<weights[i].rows; j++)
            {
                for(int k=0; k<weights[i].columns; k++)
                {
                    weights[i][j, k] = Operator<T>.multiply(weightRate, Operator<T>.multiply(layers[i - 1][k], sigmas[i][j]));
                }
            }*/
        }
    }

    public void Backpropagate(Vector<T>[] inputs, Func<Vector<T>, Vector<T>, Vector<T>> ErrorFunctionDerivative, T weightRate, T biasRate) //loops through all inputs and backpropagates
    {
        Vector<T> output;
        for (int i = 0; i < inputs.Length; i++)
        {
            output = Evaluate(inputs[i]);
            Vector<T>[] deltas = CalculateSigma(ErrorFunctionDerivative);
            UpdateWeightsBiases(deltas, weightRate, biasRate);
        }
    }


}

public static class NeuralNetworkUtil
{
    public static double Identity (double x)
    {
        return x;
    }
    public static double IdentityDerivative(double x)
    {
        return 1;
    }

    //Here we have a list of activation Functions
    public static T Tanh<T>(T x)
    {
        return Operator<double, T>.convert(Math.Tanh(Operator<T, double>.convert(x))); //we convert from T to double to evaluate tanh then convert back
    }

    public static T ATan<T>(T x)
    {
        return Operator<double, T>.convert(Math.Atan(Operator<T, double>.convert(x)));
    }

    public static T Pow<T>(T x, double y)
    {
        return Operator<double, T>.power(x, y);
    }

    public static double PiecewiseLinearUnit(double x, double c = 1, double a = 0.5)
    {
        if (x > c) return a * (x - c) + c;
        if (x < -c) return a * (x + c) - c;
        return x;
    }

    public static double InversePowUnit(double x, double a=1.0, double p=2)
    {
        return x * Math.Pow(1 + a * Math.Pow(x, p), -1.0 / p);
    }

    public static double SquareNonlinearUnit(double x, double c = 1, double a=0.1)
    {
        //we shouldn't have a<0
        double d = (a - 1.0) / 2.0;
        if (x >= c) return a * (x - c) + c * (1.0 + d);
        if (x <= -c) return a * (x + c) - c * (1.0 + d);
        if (x >= 0) return (1.0 + (d / c) * x) * x;
        return (1.0 - (d / c) * x) * x; // -c < x < 0
    }

    public static double Logistic(double x)
    {
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    //derivatives
    public static double PiecewiseLinearUnitDerivative(double x, double c=1, double a = 0.5)
    {
        if (x > c) return a;
        if (x < -c) return a;
        return 1;
    }

    public static double LogisticDerivative(double x)
    {
        return Logistic(x) * (1.0 - Logistic(x));
    }

    public static double TanhDerivative(double x)
    {
        return 1.0 - Tanh(x);
    }

    public static double InversePowUnitDerivative(double x, double a=1.0, double p=2)
    {
        return Math.Pow(1.0 + a * Math.Pow(x, p), -(1.0 / p) - 1.0);
    }

    public static double SquareNonLinearUnitDerivative(double x, double c=1, double a=0.1)
    {
        if (x >= c || x <= -c) return a;

        double d = (a - 1.0) / 2.0;
        if (x > 0) return 1.0 + 2.0 * (d / c) * x;
        return 1.0 - 2.0 * (d / c) * x;
    }

    public static double RectifiedLinear(double x) //ReLu
    {
        if (x > 0) return x;
        return 0;
    }
    public static double RectifiedLinearDerivative(double x) //ReLu
    {
        if (x > 0) return 1;
        return 0;
    }


    //Error functions, These technically should just be a norm

    public static double PNorm(Vector<double> x, double p)
    {
        double sum = 0;
        for(int i=0; i<x.rows; i++)
        {
            sum += Math.Pow(Math.Abs(x[i]), p);
        }
        return Math.Pow(sum, 1 / p);
    }
    public static double OneNorm(Vector<double> x)
    {
        return PNorm(x, 1.0);
    }
    public static double EuclidNorm(Vector<double> x)
    {
        return PNorm(x, 2.0);
    }
    public static double InfNorm(Vector<double> x) //or max norm
    {
        if (x.rows == 1) return Math.Abs(x[0]);
        double max = Math.Abs(x[0]);
        for(int i=1; i<x.rows; i++)
        {
            max = Math.Max(max, Math.Abs(x[i]));
        }
        return max;
    }


    //creates a new vector function by applying the simple function on each element of the vector
    public static Func<Vector<T>, Vector<T>> CreateVectorFunction<T>(Func<T, T> f) where T : new()
    {
        Vector<T> F(Vector<T> v)
        {
            Vector<T> u = new Vector<T>(v.rows);
            for(int i=0; i<v.rows; i++)
            {
                u[i] = f(v[i]);
            }
            return u;
        }
        return F;
    }


    public static void RandomiseArray<T>(T[] array, int seed = 0)
    {
        int n = array.Length;
        Random rand;
        if (seed == 0) { rand = new Random(); }
        else { rand = new Random(seed); }

        for(int i=0; i<n; i++)
        {
            int r = rand.Next(0, n);
            if (r == i) continue;

            T temp = array[r]; //swap r and i
            array[r] = array[i];
            array[i] = temp;
        }
    }

}
