#include <iostream>
#include <vector>
#include <cmath>


using namespace std;

int funcActivationEscalon(double x) {
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

    cout << "\n === Iniciando entrenamiento ===" << endl;
    cout << "- Pesos iniciales: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << endl;
    cout << "- Tasa de aprendizaje: " << alpha << endl;

    // Epocas de entrenamiento
    for(int epoch = 0; epoch < epochs; epoch++) {
        int totalError = 0;

        // Iterar sobre cada ejemplo de entrenamiento
        for (size_t i = 0; i < tInputs.size(); i++) {
            // Calcular la salida del perceptrón
            double x1 = tInputs[i][0]; 
            double x2 = tInputs[i][1]; 
            double z = w1 * x1 + w2 * x2 + b;
            
            // cout << "\n- Entrada: x1 = " << x1 << ",  x2 = " << x2 << endl;
            // cout << "- Pesos: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b ;
            
            // Prdecir la salida
            int predictOutput = funcActivationEscalon(z); // salidad obtenida
            
            // Calcular el error
            double error = tOutputs[i] - predictOutput; // salida esperada - salida obtenida
            
            // Actualizar los pesos y bias
            if(error != 0)
            {
                totalError++;
                w1 += alpha * error * x1;
                w2 += alpha * error * x2;
                b += alpha * error;
            }

        }
        // Mostrar progreso de la epoca
        cout << "\nEpoca: " << epoch << " - Total Error: " << totalError << endl;
        cout << "- Pesos: w1 = " << w1 << ", w2 = " << w2 << ", b = " << b << endl;
        
        // Criterio parada: Se alcanzó convergencia
        if (totalError == 0) {
            cout << "\n* Converge en la epoca: " << epoch << endl;
            return;
        }
    }
    cout << "\n* Entrenamiento finalizado ===" << endl;
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

    // Entrenar perceptron
    train(tInputs, tOutputs, w1, w2, b, alpha, epochs);
    
    // Probar el perceptrón
    cout << "\n== Prueba del perceptrón ===" << endl;
    for(int i = 0; i < tInputs.size(); i++) {
        double x1 = tInputs[i][0]; 
        double x2 = tInputs[i][1]; 
        double z = w1 * x1 + w2 * x2 + b;
        int predictOutput = funcActivationEscalon(z);
        cout << "Entrada: (" << x1 << ", " << x2 << ") \n- Salida esperada: " << tOutputs[i] << " - Salida obtenida: " << predictOutput << endl;
    }

    return 0;
}