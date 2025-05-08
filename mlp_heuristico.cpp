// PROBLEMA XOR
// ==================
// Método               : Heurístico
// Arquitectura         : 2 entradas, 2 neuronas ocultas, 1 salida
// Función de activación: sigmoide o escalón
// Función de error     : Error Cuadrático Medio (MSE) con fitnes de 4

#include <iostream> // Para entrada/salida (cout, endl)
#include <vector>   // Para std::vector (usado para representar arrays dinámicos de pesos/sesgos y datos)
#include <cmath>    // Para funciones matemáticas como abs() o pow()
#include <random>   // Para generación de números aleatorios
#include <limits>   // Para numeric_limits (para el valor inicial del mejor error)

using namespace std;

// --- Constantes de la Red Neuronal ---
const int NUM_ENTRADAS = 2;         // Número de neuronas en la capa de entrada
const int NUM_NEURONAS_OCULTAS = 2; // Número de neuronas en la capa oculta
const int NUM_SALIDAS = 1;          // Número de neuronas en la capa de salida

// --- Constantes de la Búsqueda Aleatoria ---
const double RANGO_PESOS_MIN = -5.0;   // Rango mínimo para los pesos y sesgos aleatorios
const double RANGO_PESOS_MAX = 5.0;    // Rango máximo para los pesos y sesgos aleatorios
const long long MAX_ITERACIONES = 5000000; // Número máximo de intentos para encontrar la solución

// --- Datos de Entrenamiento para XOR ---
// Entradas: {x1, x2}
// Salidas Esperadas: {y}
// Usamos arrays C-style para cumplir la restricción de 'no clases' de forma más estricta,
// aunque std::vector se usa para almacenar los pesos/sesgos generados.
const double XOR_ENTRADAS[4][NUM_ENTRADAS] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

const double XOR_SALIDAS_ESPERADAS[4] = {
    0.0,
    1.0,
    1.0,
    0.0
};

// --- Funciones Auxiliares (permitidas por la consigna) ---

// Función de Activación (escalón unitario)
// Devuelve 1.0 si la suma ponderada es >= 0.0, de lo contrario 0.0
double step_activation(double sum) {
    if (sum >= 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// Función para calcular el Error Cuadrático Medio (MSE)
// predicted_outputs y expected_outputs son arrays C-style.
double calculate_mse(const double predicted_outputs[], const double expected_outputs[], int num_samples) {
    double mse = 0.0;
    for (int i = 0; i < num_samples; ++i) {
        double error = expected_outputs[i] - predicted_outputs[i];
        mse += error * error;
    }
    return mse / num_samples;
}

// --- Función Principal (main) ---
int main() {
    // --- Configuración para generación de números aleatorios ---
    random_device rd;   // Semilla para el generador
    mt19937 gen(rd());  // Generador de números pseudoaleatorios Mersenne Twister
    // Distribución uniforme para generar números reales dentro del rango de pesos
    uniform_real_distribution<> dis(RANGO_PESOS_MIN, RANGO_PESOS_MAX);

    // --- Variables para almacenar la mejor red encontrada ---
    // Pesos de entrada a capa oculta (NUM_ENTRADAS x NUM_NEURONAS_OCULTAS)
    vector<vector<double>> best_weights_ih(NUM_ENTRADAS, vector<double>(NUM_NEURONAS_OCULTAS));
    // Sesgos de la capa oculta (NUM_NEURONAS_OCULTAS)
    vector<double> best_biases_h(NUM_NEURONAS_OCULTAS);
    // Pesos de capa oculta a salida (NUM_NEURONAS_OCULTAS x NUM_SALIDAS)
    vector<double> best_weights_ho(NUM_NEURONAS_OCULTAS);
    // Sesgo de la capa de salida (NUM_SALIDAS, en este caso 1)
    double best_bias_o;
    // Mejor error cuadrático medio encontrado (inicializado a un valor muy grande)
    double best_mse = numeric_limits<double>::max();

    // --- Variables temporales para la red actual en cada iteración ---
    vector<vector<double>> current_weights_ih(NUM_ENTRADAS, vector<double>(NUM_NEURONAS_OCULTAS));
    vector<double> current_biases_h(NUM_NEURONAS_OCULTAS);
    vector<double> current_weights_ho(NUM_NEURONAS_OCULTAS);
    double current_bias_o;

    // --- Bucle Principal de Búsqueda Aleatoria ---
    cout << "Iniciando Búsqueda Aleatoria para XOR..." << endl;
    cout << "------------------------------------" << endl;

    for (long long iter = 0; iter < MAX_ITERACIONES; ++iter) {
        // 1. Generar pesos y sesgos aleatorios para la red actual
        for (int i = 0; i < NUM_ENTRADAS; ++i) {
            for (int j = 0; j < NUM_NEURONAS_OCULTAS; ++j) {
                current_weights_ih[i][j] = dis(gen);
            }
        }
        for (int i = 0; i < NUM_NEURONAS_OCULTAS; ++i) {
            current_biases_h[i] = dis(gen);
            current_weights_ho[i] = dis(gen);
        }
        current_bias_o = dis(gen);

        // 2. Calcular las salidas de la red para cada dato de entrenamiento XOR
        double current_predicted_outputs[4]; // Array para almacenar las 4 salidas predichas

        for (int i = 0; i < 4; ++i) { // Iterar sobre los 4 ejemplos de XOR
            const double* input_sample = XOR_ENTRADAS[i];

            // Propagación hacia adelante: Capa Oculta
            double hidden_outputs[NUM_NEURONAS_OCULTAS];
            for (int h = 0; h < NUM_NEURONAS_OCULTAS; ++h) {
                double sum_hidden = 0.0;
                for (int j = 0; j < NUM_ENTRADAS; ++j) {
                    sum_hidden += input_sample[j] * current_weights_ih[j][h];
                }
                sum_hidden += current_biases_h[h];
                hidden_outputs[h] = step_activation(sum_hidden);
            }

            // Propagación hacia adelante: Capa de Salida
            double sum_output = 0.0;
            for (int h = 0; h < NUM_NEURONAS_OCULTAS; ++h) {
                sum_output += hidden_outputs[h] * current_weights_ho[h];
            }
            sum_output += current_bias_o;
            current_predicted_outputs[i] = step_activation(sum_output);
        }

        // 3. Calcular el Error Cuadrático Medio (MSE) para la red actual
        double current_mse = calculate_mse(current_predicted_outputs, XOR_SALIDAS_ESPERADAS, 4);

        // 4. Actualizar el mejor resultado si la red actual es mejor
        if (current_mse < best_mse) {
            best_mse = current_mse;

            // Copiar los pesos y sesgos de la red actual a la "mejor red"
            for (int i = 0; i < NUM_ENTRADAS; ++i) {
                for (int j = 0; j < NUM_NEURONAS_OCULTAS; ++j) {
                    best_weights_ih[i][j] = current_weights_ih[i][j];
                }
            }
            for (int i = 0; i < NUM_NEURONAS_OCULTAS; ++i) {
                best_biases_h[i] = current_biases_h[i];
                best_weights_ho[i] = current_weights_ho[i];
            }
            best_bias_o = current_bias_o;

            // Imprimir progreso
            cout << "Iteración: " << iter + 1 << " - Nuevo mejor MSE: " << best_mse << endl;

            // Criterio de parada: si se encuentra una solución perfecta (MSE ~ 0)
            if (best_mse == 0.0) {
                cout << "¡XOR resuelto! Encontrada una red perfecta." << endl;
                break; // Salir del bucle principal
            }
        }
    }

    cout << "\n------------------------------------" << endl;
    cout << "Búsqueda Aleatoria Finalizada." << endl;
    cout << "Mejor Error Cuadrático Medio Obtenido: " << best_mse << endl;

    // --- Imprimir la mejor red encontrada ---
    cout << "\nMejor Red Encontrada (para XOR):" << endl;

    cout << "  Pesos Entrada -> Oculta (W_IH):" << endl;
    for (int i = 0; i < NUM_ENTRADAS; ++i) {
        cout << "    [";
        for (int j = 0; j < NUM_NEURONAS_OCULTAS; ++j) {
            cout << best_weights_ih[i][j] << (j == NUM_NEURONAS_OCULTAS - 1 ? "" : ", ");
        }
        cout << "]" << endl;
    }

    cout << "  Sesgos Oculta (B_H): [";
    for (int i = 0; i < NUM_NEURONAS_OCULTAS; ++i) {
        cout << best_biases_h[i] << (i == NUM_NEURONAS_OCULTAS - 1 ? "" : ", ");
    }
    cout << "]" << endl;

    cout << "  Pesos Oculta -> Salida (W_HO): [";
    for (int i = 0; i < NUM_NEURONAS_OCULTAS; ++i) {
        cout << best_weights_ho[i] << (i == NUM_NEURONAS_OCULTAS - 1 ? "" : ", ");
    }
    cout << "]" << endl;

    cout << "  Sesgo Salida (B_O): " << best_bias_o << endl;

    // --- Verificar el funcionamiento de la mejor red encontrada ---
    cout << "\nVerificación de la Mejor Red Encontrada:" << endl;
    for (int i = 0; i < 4; ++i) {
        const double* input_sample = XOR_ENTRADAS[i];
        double expected_output = XOR_SALIDAS_ESPERADAS[i];

        // Propagación hacia adelante: Capa Oculta (usando la MEJOR_RED)
        double hidden_outputs[NUM_NEURONAS_OCULTAS];
        for (int h = 0; h < NUM_NEURONAS_OCULTAS; ++h) {
            double sum_hidden = 0.0;
            for (int j = 0; j < NUM_ENTRADAS; ++j) {
                sum_hidden += input_sample[j] * best_weights_ih[j][h];
            }
            sum_hidden += best_biases_h[h];
            hidden_outputs[h] = step_activation(sum_hidden);
        }

        // Propagación hacia adelante: Capa de Salida (usando la MEJOR_RED)
        double sum_output = 0.0;
        for (int h = 0; h < NUM_NEURONAS_OCULTAS; ++h) {
            sum_output += hidden_outputs[h] * best_weights_ho[h];
        }
        sum_output += best_bias_o;
        double predicted_output = step_activation(sum_output);

        cout << "  Entrada: (" << input_sample[0] << ", " << input_sample[1]
             << ") -> Predicción: " << predicted_output
             << ", Esperado: " << expected_output
             << " -> " << (predicted_output == expected_output ? "Correcto" : "Incorrecto") << endl;
    }

    return 0;
}

