#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float sigmoid_derivative(float x) {
    float s = sigmoid(x);
    return s * (1.0f - s);
}

__global__ void forward_kernel(
    float* inputs, float* weights_ih, float* bias_h, float* hidden_outputs,
    float* weights_ho, float* bias_o, float* final_outputs,
    float* hidden_inputs, float* final_inputs) {

    int idx = threadIdx.x;

    if (idx < HIDDEN_SIZE) {
        float sum = bias_h[idx];
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += inputs[i] * weights_ih[i * HIDDEN_SIZE + idx];
        }
        hidden_inputs[idx] = sum;
        hidden_outputs[idx] = sigmoid(sum);
    }

    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        float sum = bias_o[idx];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += hidden_outputs[j] * weights_ho[j * OUTPUT_SIZE + idx];
        }
        final_inputs[idx] = sum;
        final_outputs[idx] = sigmoid(sum);
    }
}

__global__ void backpropagation_kernel(
    float* inputs, float* targets, float* hidden_outputs, float* final_outputs,
    float* final_inputs, float* hidden_inputs, float* weights_ho, float* weights_ih,
    float* bias_h, float* bias_o, float learning_rate) {

    __shared__ float output_deltas[OUTPUT_SIZE];
    __shared__ float hidden_deltas[HIDDEN_SIZE];

    int idx = threadIdx.x;

    if (idx < OUTPUT_SIZE) {
        float error = targets[idx] - final_outputs[idx];
        output_deltas[idx] = error * sigmoid_derivative(final_inputs[idx]);
    }

    __syncthreads();

    if (idx < HIDDEN_SIZE) {
        float error = 0.0f;
        for (int k = 0; k < OUTPUT_SIZE; ++k) {
            error += output_deltas[k] * weights_ho[idx * OUTPUT_SIZE + k];
        }
        hidden_deltas[idx] = error * sigmoid_derivative(hidden_inputs[idx]);
    }

    __syncthreads();

    if (idx < OUTPUT_SIZE) {
        bias_o[idx] += learning_rate * output_deltas[idx];
        for (int j = 0; j < HIDDEN_SIZE; ++j) {
            weights_ho[j * OUTPUT_SIZE + idx] += learning_rate * output_deltas[idx] * hidden_outputs[j];
        }
    }

    if (idx < HIDDEN_SIZE) {
        bias_h[idx] += learning_rate * hidden_deltas[idx];
        for (int i = 0; i < INPUT_SIZE; ++i) {
            weights_ih[i * HIDDEN_SIZE + idx] += learning_rate * hidden_deltas[idx] * inputs[i];
        }
    }
}

void random_init(float* array, int size) {
    for (int i = 0; i < size; ++i)
        array[i] = ((float)rand() / RAND_MAX) * 0.2f - 0.1f;
}

void load_data_from_csv(const std::string& file_path, std::vector<std::vector<float>>& data, std::vector<std::vector<float>>& labels, int limit = 1000) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        std::cerr << "No se pudo abrir el archivo: " << file_path << std::endl;
        return;
    }

    std::string line;
    std::getline(file, line); 

    int count = 0;
    while (std::getline(file, line) && count < limit) {
        std::stringstream ss(line);
        std::string token;

        std::getline(ss, token, ',');
        int label = std::stoi(token);
        labels.push_back(std::vector<float>(OUTPUT_SIZE, 0.0f));
        labels.back()[label] = 1.0f;

        std::vector<float> row(INPUT_SIZE);
        for (int i = 0; i < INPUT_SIZE; i++) {
            std::getline(ss, token, ',');
            row[i] = std::stof(token) / 255.0f;
        }
        data.push_back(row);
        count++;
    }
}

int main() {
    srand(time(0));
    std::vector<std::vector<float>> train_data, train_labels;
    load_data_from_csv("Data/mnist_train.csv", train_data, train_labels, 1000);

    float* h_weights_ih = new float[INPUT_SIZE * HIDDEN_SIZE];
    float* h_weights_ho = new float[HIDDEN_SIZE * OUTPUT_SIZE];
    float* h_bias_h = new float[HIDDEN_SIZE];
    float* h_bias_o = new float[OUTPUT_SIZE];

    random_init(h_weights_ih, INPUT_SIZE * HIDDEN_SIZE);
    random_init(h_weights_ho, HIDDEN_SIZE * OUTPUT_SIZE);
    random_init(h_bias_h, HIDDEN_SIZE);
    random_init(h_bias_o, OUTPUT_SIZE);

    float* d_input, * d_target, * d_weights_ih, * d_weights_ho, * d_bias_h, * d_bias_o;
    float* d_hidden_outputs, * d_final_outputs, * d_hidden_inputs, * d_final_inputs;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_target, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_weights_ih, INPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_weights_ho, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_bias_h, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_bias_o, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_outputs, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_final_outputs, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_hidden_inputs, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_final_inputs, OUTPUT_SIZE * sizeof(float));

    cudaMemcpy(d_weights_ih, h_weights_ih, INPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights_ho, h_weights_ho, HIDDEN_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_h, h_bias_h, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_o, h_bias_o, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    float lr = 0.1f;
    std::vector<float> output(OUTPUT_SIZE);

    std::ofstream log_file("learning_curve.csv");
    log_file << "epoch,error,accuracy\n";

    for (int epoch = 0; epoch < 10; epoch++) {
        float total_error = 0.0f;
        int correct = 0;

        for (size_t i = 0; i < train_data.size(); i++) {
            cudaMemcpy(d_input, train_data[i].data(), INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_target, train_labels[i].data(), OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

            forward_kernel << <1, 128 >> > (d_input, d_weights_ih, d_bias_h, d_hidden_outputs,
                d_weights_ho, d_bias_o, d_final_outputs,
                d_hidden_inputs, d_final_inputs);

            backpropagation_kernel << <1, 128 >> > (d_input, d_target, d_hidden_outputs, d_final_outputs,
                d_final_inputs, d_hidden_inputs,
                d_weights_ho, d_weights_ih,
                d_bias_h, d_bias_o, lr);

            cudaMemcpy(output.data(), d_final_outputs, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

            float error = 0.0f;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                float e = train_labels[i][j] - output[j];
                error += 0.5f * e * e;
            }
            total_error += error;

            int predicted = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
            int true_label = std::distance(train_labels[i].begin(), std::max_element(train_labels[i].begin(), train_labels[i].end()));

            if (predicted == true_label) correct++;
        }

        float avg_error = total_error / train_data.size();
        float accuracy = (float)correct / train_data.size() * 100.0f;

        std::cout << "Epoch " << (epoch + 1)
            << " | Error: " << avg_error
            << " | Accuracy: " << accuracy << "%" << std::endl;

        log_file << (epoch + 1) << "," << avg_error << "," << accuracy << "\n";
    }

    log_file.close();

    cudaFree(d_input); cudaFree(d_target); cudaFree(d_weights_ih); cudaFree(d_weights_ho);
    cudaFree(d_bias_h); cudaFree(d_bias_o); cudaFree(d_hidden_outputs); cudaFree(d_final_outputs);
    cudaFree(d_hidden_inputs); cudaFree(d_final_inputs);

    delete[] h_weights_ih; delete[] h_weights_ho; delete[] h_bias_h; delete[] h_bias_o;

    // Ejecutar script de Python para graficar
    int status = system("\"C:\\Users\\user\\AppData\\Local\\Programs\\Python\\Python313\\python.exe\" plot_learning_curve.py");
    std::cout << "Python script executed with code: " << status << std::endl;



    return 0;
}
