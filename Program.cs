using System;
using LinearAlgebra;

namespace MonoNerualNetwork{
    /*
        👀 REDES NEURONALES en 100 lineas DESDE CERO ► [MATRICES para redes neuronales] - Hector Pulido
        https://youtu.be/HRYYxJd9qiA
    */
    class Program{
        static void Main(string[] args){

            int epochs = 2000;

            int inputCount = 2;
            int outputCount = 1;
            int hiddenCount = 5; //Neruonas
            int examplesCount = 4;
            double learningRate = 0.2;

            Matrix x = new double[,] {  {0, 0}, 
                                        {0, 1}, 
                                        {1, 0},
                                        {1, 1}};

            Matrix y = new double[,] {  {1}, 
                                        {0}, 
                                        {0},
                                        {1}};

            Random r = new Random(1);

            Matrix w1 = (Matrix.Random(inputCount + 1, hiddenCount, r) * 2) -1;
            Matrix w2 = (Matrix.Random(hiddenCount + 1, outputCount, r) * 2) -1;

            Console.WriteLine("-------Entrada-------");
            Console.WriteLine(x);

            for (int epoch = 0; epoch <= epochs + 1; epoch++){
                //FORWARDPROPAGATION
                Matrix z1 = x.AddColumn(Matrix.Ones(examplesCount, 1));
                Matrix a1 = z1; // Primera capa (Entrada)

                Matrix z2 = (a1 * w1).AddColumn(Matrix.Ones(examplesCount, 1));
                Matrix a2 = Sigmoid(z2); // Capa oculta

                Matrix z3 = (a2 * w2);
                Matrix a3 = Sigmoid(z3); // Capa de salida

                //BACKPROPAGATION
                Matrix Error3 = a3 - y;
                Matrix Delta3 = Error3 * Sigmoid(z3, true);

                Matrix Error2 = Delta3 * w2.T;
                Matrix Delta2 = Error2 * Sigmoid(z2, true);
                Delta2 = Delta2.Slice(0, 1, Delta2.x, Delta2.y);

                w2 -= (a2.T * Delta3) * learningRate;
                w1 -= (a1.T * Delta2) * learningRate;

                //Console.WriteLine(Error3.abs.average * examplesCount);

                if (epoch % 1000 == 0){
                    Console.WriteLine("-------" + epoch + "-------");
                    Console.WriteLine("Salida:\n" + a3);
                }
            }

        }

        static Matrix Sigmoid(Matrix input, bool derivated = false){
            double[,] output = input;
            Matrix.MatrixLoop((i, j) => {
                if (derivated){
                    output[i, j] = 1 / (1 + Math.Exp(- output[i, j]));
                    output[i, j] = output[i, j] * (1 - output[i, j]);
                }else{
                    output[i, j] = 1 / (1 + Math.Exp(- output[i, j]));
                }
            }, input.x, input.y);
            return output;
        }
    }
}