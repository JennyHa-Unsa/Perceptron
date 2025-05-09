#include <iostream>
#include <cstdlib>  // Para rand(), srand()
#include <ctime>    // Para time()
#include <limits>   // Para numeric_limits

using namespace std;

// Constantes de la Red Neuronal 
const int NUM_ENTRADAS = 2;         // Nro. de neuronas en la capa de entrada
const int NUM_NEURONAS_OCULTAS = 2; // Nro. de neuronas en la capa oculta
const int NUM_SALIDAS = 1;          // Nro. de neuronas en la capa de salida

// Constantes de la Búsqueda Aleatoria 
const double RANGO_PESOS_MIN = -5.0;   // Rango mínimo para los pesos y sesgos aleatorios
const double RANGO_PESOS_MAX = 5.0;    // Rango máximo para los pesos y sesgos aleatorios
const int MAX_ITERACIONES = 5000000; // Nro. máximo de intentos para encontrar la solución

// Datos de Entrenamiento para XOR 
const double DATOS_ENTRADA_XOR[4][NUM_ENTRADAS] = {
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
    {1.0, 1.0}
};

const double SALIDAS_ESPERADAS_XOR[4] = {
    0.0,
    1.0,
    1.0,
    0.0
};



// Función de Activación (escalón unitario)
double activacionEscalon(double sumaPonderada) {
    if (sumaPonderada >= 0.0) {
        return 1.0;
    } else {
        return 0.0;
    }
}

// Función para calcular el Error Cuadrático Medio (MSE)
double calcularMSE(const double salidasPredichas[], const double salidasEsperadas[], int numDatos) {
    double mse = 0.0;
    for (int i = 0; i < numDatos; i++) {
        double error = salidasEsperadas[i] - salidasPredichas[i];
        mse += error * error;
    }
    return mse / numDatos;
}

// Función de Búsqueda Aleatoria 
// - Encuentra los mejores pesos y sesgos para resolver XOR usando búsqueda aleatoria
void ejecutarBusquedaAleatoria(
    double mejoresPesosEntradaOculta[NUM_ENTRADAS][NUM_NEURONAS_OCULTAS], 
    double mejoresSesgosOculta[NUM_NEURONAS_OCULTAS],
    double mejoresPesosOcultaSalida[NUM_NEURONAS_OCULTAS],
    double& mejorSesgoSalida,       // Por referencia para modificar el valor original
    double& mejorMseEncontrado      // Por referencia para modificar el valor original
) {
    // Sembrar el generador de números aleatorios
    srand(static_cast<unsigned int>(time(0)));

    // Variables temporales para la red actual en cada iteración
    double pesosEntradaOcultaActual[NUM_ENTRADAS][NUM_NEURONAS_OCULTAS];
    double sesgosOcultaActual[NUM_NEURONAS_OCULTAS];
    double pesosOcultaSalidaActual[NUM_NEURONAS_OCULTAS];
    double sesgoSalidaActual;

    // Almacena el mejor MSE encontrado internamente
    double mejorMseInterno = numeric_limits<double>::max();

    cout << "\nIniciando Búsqueda Aleatoria para XOR" << endl;
    cout << "------------------------------------" << endl;

    for (int iteracion = 0; iteracion < MAX_ITERACIONES; iteracion++) {

        // 1. Generar pesos y sesgos aleatorios para la red actual
        for (int i = 0; i < NUM_ENTRADAS; i++) {
            for (int j = 0; j < NUM_NEURONAS_OCULTAS; j++) {
                pesosEntradaOcultaActual[i][j] = RANGO_PESOS_MIN + (double)rand() / RAND_MAX * (RANGO_PESOS_MAX - RANGO_PESOS_MIN);
            }
        }
        for (int i = 0; i < NUM_NEURONAS_OCULTAS; i++) {
            sesgosOcultaActual[i] = RANGO_PESOS_MIN + (double)rand() / RAND_MAX * (RANGO_PESOS_MAX - RANGO_PESOS_MIN);
            pesosOcultaSalidaActual[i] = RANGO_PESOS_MIN + (double)rand() / RAND_MAX * (RANGO_PESOS_MAX - RANGO_PESOS_MIN);
        }
        sesgoSalidaActual = RANGO_PESOS_MIN + (double)rand() / RAND_MAX * (RANGO_PESOS_MAX - RANGO_PESOS_MIN);

       
        // 2. Calcular las salidas de la red para cada dato de entrenamiento XOR (4 salidas predichas)
        double salidasPredichasActuales[4]; 

        for (int i = 0; i < 4; i++) { // Iterar sobre los 4 ejemplos de XOR
            const double* ejemploEntrada = DATOS_ENTRADA_XOR[i];

            // Propagación hacia adelante: Capa Oculta
            double salidasOcultas[NUM_NEURONAS_OCULTAS];
            for (int h = 0; h < NUM_NEURONAS_OCULTAS; h++) {
                double sumaPonderadaOculta = 0.0;
                for (int j = 0; j < NUM_ENTRADAS; j++) {
                    sumaPonderadaOculta += ejemploEntrada[j] * pesosEntradaOcultaActual[j][h];
                }
                sumaPonderadaOculta += sesgosOcultaActual[h];
                salidasOcultas[h] = activacionEscalon(sumaPonderadaOculta);
            }

            // Propagación hacia adelante: Capa de Salida
            double sumaPonderadaSalida = 0.0;
            for (int h = 0; h < NUM_NEURONAS_OCULTAS; h++) {
                sumaPonderadaSalida += salidasOcultas[h] * pesosOcultaSalidaActual[h];
            }
            sumaPonderadaSalida += sesgoSalidaActual;
            salidasPredichasActuales[i] = activacionEscalon(sumaPonderadaSalida);
        }

        // 3. Calcular el Error Cuadrático Medio (MSE) para la red actual
        double errorCuadraticoMedioActual = calcularMSE(salidasPredichasActuales, SALIDAS_ESPERADAS_XOR, 4);

        // 4. Actualizar el mejor resultado si la red actual es mejor
        if (errorCuadraticoMedioActual < mejorMseInterno) {
            mejorMseInterno = errorCuadraticoMedioActual;

            // Copiar los pesos y sesgos de la red actual a los arrays de la "mejor red"
            for (int i = 0; i < NUM_ENTRADAS; i++) {
                for (int j = 0; j < NUM_NEURONAS_OCULTAS; j++) {
                    mejoresPesosEntradaOculta[i][j] = pesosEntradaOcultaActual[i][j];
                }
            }
            for (int i = 0; i < NUM_NEURONAS_OCULTAS; i++) {
                mejoresSesgosOculta[i] = sesgosOcultaActual[i];
                mejoresPesosOcultaSalida[i] = pesosOcultaSalidaActual[i];
            }
            mejorSesgoSalida = sesgoSalidaActual; // Asignación directa a la referencia

            // Imprimir progreso
            cout << "Iteración: " << iteracion + 1 << " - Nuevo mejor MSE: " << mejorMseInterno << endl;

            // Criterio de parada: si se encuentra una solución perfecta (MSE == 0.0)
            if (mejorMseInterno == 0.0) {
                cout << "¡XOR resuelto! Se encontró una red perfecta." << endl;
                break; // Salir del bucle principal de ITERACIONES
            }
        }
    }

    // Almacenar el mejor MSE encontrado en el parámetro de salida
    mejorMseEncontrado = mejorMseInterno; // Asignación directa a la referencia
}


int main() {
    // Arrays para almacenar la mejor red encontrada por la función de búsqueda
    double pesosFinalesEntradaOculta[NUM_ENTRADAS][NUM_NEURONAS_OCULTAS];
    double sesgosFinalesOculta[NUM_NEURONAS_OCULTAS];
    double pesosFinalesOcultaSalida[NUM_NEURONAS_OCULTAS];
    double sesgoFinalSalida;
    double mseFinalMejor;

    // Ejecutar la búsqueda aleatoria, pasando las variables escalares por referencia
    ejecutarBusquedaAleatoria( 
        pesosFinalesEntradaOculta,
        sesgosFinalesOculta,
        pesosFinalesOcultaSalida,
        sesgoFinalSalida,     // Por referencia
        mseFinalMejor         // Por referencia
    );

    cout << "\nBúsqueda Aleatoria Finalizada." << endl;
    cout << "------------------------------------" << endl;
    cout << "Mejor Error Cuadrático Medio Obtenido: " << mseFinalMejor << endl;

    // Imprimir la mejor red encontrada 
    cout << "\nMejor Red Encontrada (para XOR):" << endl;

    cout << "- Pesos Entrada -> Oculta (W_IH):" << endl;
    for (int i = 0; i < NUM_ENTRADAS; i++) {
        cout << "\t[";
        for (int j = 0; j < NUM_NEURONAS_OCULTAS; j++) {
            cout << pesosFinalesEntradaOculta[i][j] << (j == NUM_NEURONAS_OCULTAS - 1 ? "" : ", ");
        }
        cout << "]" << endl;
    }

    cout << "- Sesgos Oculta (B_H):\n\t[";
    for (int i = 0; i < NUM_NEURONAS_OCULTAS; i++) {
        cout << sesgosFinalesOculta[i] << (i == NUM_NEURONAS_OCULTAS - 1 ? "" : ", ");
    }
    cout << "]" << endl;

    cout << "- Pesos Oculta -> Salida (W_HO):\n\t[";
    for (int i = 0; i < NUM_NEURONAS_OCULTAS; i++) {
        cout << pesosFinalesOcultaSalida[i] << (i == NUM_NEURONAS_OCULTAS - 1 ? "" : ", ");
    }
    cout << "]" << endl;

    cout << "- Sesgo Salida (B_O):\n\t" << sesgoFinalSalida << endl;

    // Verificar el funcionamiento de la mejor red encontrada 
    cout << "\nVerificación de la Mejor Red Encontrada:" << endl;
    for (int i = 0; i < 4; i++) {
        const double* ejemploEntrada = DATOS_ENTRADA_XOR[i];
        double salidaEsperada = SALIDAS_ESPERADAS_XOR[i];

        // Propagación hacia adelante: Capa Oculta (usando la MEJOR_RED)
        double salidasOcultas[NUM_NEURONAS_OCULTAS];
        for (int h = 0; h < NUM_NEURONAS_OCULTAS; h++) {
            double sumaPonderadaOculta = 0.0;
            for (int j = 0; j < NUM_ENTRADAS; j++) {
                sumaPonderadaOculta += ejemploEntrada[j] * pesosFinalesEntradaOculta[j][h];
            }
            sumaPonderadaOculta += sesgosFinalesOculta[h];
            salidasOcultas[h] = activacionEscalon(sumaPonderadaOculta);
        }

        // Propagación hacia adelante: Capa de Salida (usando la MEJOR_RED)
        double sumaPonderadaSalida = 0.0;
        for (int h = 0; h < NUM_NEURONAS_OCULTAS; h++) {
            sumaPonderadaSalida += salidasOcultas[h] * pesosFinalesOcultaSalida[h];
        }
        sumaPonderadaSalida += sesgoFinalSalida;
        double salidaPredicha = activacionEscalon(sumaPonderadaSalida);

        cout << "  Entrada: (" << ejemploEntrada[0] << ", " << ejemploEntrada[1]
             << ") -> Predicción: " << salidaPredicha
             << ", Esperado: " << salidaEsperada
             << " -> " << (salidaPredicha == salidaEsperada ? "Correcto" : "Incorrecto") << endl;
    }

    return 0;
}