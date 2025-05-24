#include <iostream>
#include <cuda_runtime.h>

#define FILAS 8
#define COLS 6
#define PESOS_ENTRADA 96  // 8x12
#define NUM_PERCEPTRONES 10

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "Error CUDA en " << #call << ": " << cudaGetErrorString(err) << " (Línea: " << __LINE__ << ")" << std::endl; \
        exit(1); \
    } \
}

const int digitos_cpu[10][FILAS][COLS] = {
    {{0,1,1,1,1,0}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,0}}, // 0
    {{0,0,1,1,0,0}, {0,1,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {1,1,1,1,1,1}}, // 1
    {{0,1,1,1,1,0}, {1,1,0,0,1,1}, {0,0,0,0,1,1}, {0,0,0,1,1,0}, {0,0,1,1,0,0}, {0,1,1,0,0,0}, {1,1,0,0,0,0}, {1,1,1,1,1,1}}, // 2
    {{1,1,1,1,1,0}, {0,0,0,0,1,1}, {0,0,0,1,1,0}, {0,0,1,1,0,0}, {0,0,0,1,1,0}, {0,0,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,0}}, // 3
    {{0,0,0,1,1,0}, {0,0,1,1,1,0}, {0,1,1,1,1,0}, {1,1,0,1,1,0}, {1,1,1,1,1,1}, {0,0,0,1,1,0}, {0,0,0,1,1,0}, {0,0,0,1,1,0}}, // 4
    {{1,1,1,1,1,1}, {1,1,0,0,0,0}, {1,1,1,1,1,0}, {0,0,0,0,1,1}, {0,0,0,0,1,1}, {0,0,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,0}}, // 5
    {{0,0,1,1,1,0}, {0,1,1,0,0,0}, {1,1,0,0,0,0}, {1,1,1,1,1,0}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,0}}, // 6
    {{1,1,1,1,1,1}, {0,0,0,0,1,1}, {0,0,0,1,1,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}, {0,0,1,1,0,0}}, // 7
    {{0,1,1,1,1,0}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,0}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,0}}, // 8
    {{0,1,1,1,1,0}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {1,1,0,0,1,1}, {0,1,1,1,1,1}, {0,0,0,0,1,1}, {0,0,0,1,1,0}, {0,1,1,0,0,0}}  // 9
};

__global__ void entrenar(
    float *pesos, 
    const int *entrada, 
    int salida_esperada, 
    float bias, 
    int num_perceptrones,
    int epoca_actual,
    int numero_actual
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_perceptrones) return;

    float u = bias * pesos[idx * (PESOS_ENTRADA + 1) + PESOS_ENTRADA];
    for (int i = 0; i < PESOS_ENTRADA; ++i) {
        u += entrada[i] * pesos[idx * (PESOS_ENTRADA + 1) + i];
    }
    int salida = (u > 0) ? 1 : 0;
    int error = salida_esperada - salida;

    for (int i = 0; i < PESOS_ENTRADA; ++i) {
        pesos[idx * (PESOS_ENTRADA + 1) + i] += error * entrada[i];
    }
    pesos[idx * (PESOS_ENTRADA + 1) + PESOS_ENTRADA] += error * bias;

    if (idx == numero_actual - 1) {  
        printf("[Epoca %d] Perceptron %d | Numero %d | Error: %d | Pesos: ",
               epoca_actual + 1, idx + 1, numero_actual, error);
        for (int i = 0; i < 5; ++i) {
            printf("%.1f ", pesos[idx * (PESOS_ENTRADA + 1) + i]);
        }
        printf("...\n");
    }
}

void obtenerPatronNumerico(int numero, int *matriz_flat) {
    int d1 = numero / 10;
    int d2 = numero % 10;
    if (numero < 10) d1 = 0;

    for (int i = 0; i < FILAS; ++i) {
        for (int j = 0; j < COLS; ++j) {
            matriz_flat[i * 12 + j] = digitos_cpu[d1][i][j];
            matriz_flat[i * 12 + j + 6] = digitos_cpu[d2][i][j];
        }
    }
}

int main() {
    float *h_pesos = new float[NUM_PERCEPTRONES * (PESOS_ENTRADA + 1)]();
    int *h_entrada = new int[PESOS_ENTRADA];
    float *d_pesos;
    int *d_entrada;
    cudaMalloc(&d_pesos, NUM_PERCEPTRONES * (PESOS_ENTRADA + 1) * sizeof(float));
    cudaMalloc(&d_entrada, PESOS_ENTRADA * sizeof(int));
    cudaMemcpy(d_pesos, h_pesos, NUM_PERCEPTRONES * (PESOS_ENTRADA + 1) * sizeof(float), cudaMemcpyHostToDevice);

    for (int epoca = 0; epoca < 20; ++epoca) {
        std::cout << "\nÉpoca " << epoca + 1 << "\n";
        for (int num = 1; num <= NUM_PERCEPTRONES; ++num) {
            obtenerPatronNumerico(num, h_entrada);
            cudaMemcpy(d_entrada, h_entrada, PESOS_ENTRADA * sizeof(int), cudaMemcpyHostToDevice);

            int blockSize = 256;
            int gridSize = (NUM_PERCEPTRONES + blockSize - 1) / blockSize;
            entrenar<<<gridSize, blockSize>>>(
                d_pesos, d_entrada, 
                (num == epoca + 1) ? 1 : 0,  // salida_esperada
                1.0f,                         // bias
                NUM_PERCEPTRONES,
                epoca,                        // epoca_actual
                num                           // numero_actual
            );
            cudaDeviceSynchronize();
        }
    }

    delete[] h_pesos;
    delete[] h_entrada;
    cudaFree(d_pesos);
    cudaFree(d_entrada);

    std::cout << "\nEntrenamiento completo.\n";
    return 0;
}