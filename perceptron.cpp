#include <iostream>
#include <vector>
#include <cmath>


using namespace std;

int funcionActivation(double x) {
    // Función de activación escalón
    if (x >= 0) {
        return 1;
    } else // x < 0
    {
        return 0;
    }
}

// Pesos y bicas pasados por referencia para modificar sus valores
void train (
    const vector<vector<double>>& tInputs, 
    const vector<double>& tOutputs, 
    double& w1, 
    double& w2, 
    double& b, 
    double alpha, 
    int epochs
) {

    // Epocas de entrenamiento
    for(int epoch = 0; epoch < epochs; epoch++) {
        // Iterar sobre cada entrada
        for (size_t i = 0; i < tInputs.size(); i++) {
            // Calcular la salida del perceptrón
            double z = tInputs[i][0] * w1 + tInputs[i][1] * w2 + b;
            int predictOutput = funcionActivation(z); // salidad obtenida

            // Actualizar los pesos y bias
            double error = tOutputs[i] - predictOutput; // salida desperada - salida obtenida
            w1 += alpha * error * tInputs[i][0];
            w2 += alpha * error * tInputs[i][1];
            b += alpha * error;
        }
    }
}



int main() {

    // Datos de entrenamiento
    vector<vector<double>> tInputs = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };
    vector<double> tOutputs = {
        0,
        0, 
        0, 
        1
    }; 

    // tasa de aprendizaje
    double alpha = 0.1;

    // valores iniciales de los pesos y bias
    double w1 = 0.0;
    double w2 = 0.0;
    double b = 0.0;
    
    // número máx de épocas para entrenar
    int epochs = 100;

    return 0;
}