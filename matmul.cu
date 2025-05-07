#include <math.h>
#include <stdio.h>
#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <iostream>
#include "apis_cu.h"
#include "device_launch_parameters.h"


__global__ void matmul_kernel(float* A, float* B, float* C, int M, int N, int K, int batch_size){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.z * blockDim.z + threadIdx.z;

    if(row < M && col < N && batch < batch_size){
        float sum = 0.0f;
        for(int i = 0; i < K; i++){
            sum += A[batch * M * K + row * K + i] * B[batch * K * N + i * N + col];
        }
        C[row * N + col] = sum;
    }
} 

// 辅助函数：在GPU上执行view操作
__device__ int compute_flat_index(int* idx, int* shape, int ndim) {
    int flat_idx = 0;
    int stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        flat_idx += idx[i] * stride;
        stride *= shape[i];
    }
    return flat_idx;
}
//transpose
__global__ void transpose_kernel(
    const float* input,
    float* output,
    int* shape,
    int ndim,
    int dim0,
    int dim1,
    int total_elements
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;

    // 1. 将 flat index 映射成多维索引
    int idx_nd[10];  // 假设不超过 10 维
    int temp = idx;
    int stride = 1;

    for (int i = ndim - 1; i >= 0; --i) {
        stride = 1;
        for (int j = i + 1; j < ndim; ++j)
            stride *= shape[j];
        idx_nd[i] = temp / stride;
        temp %= stride;
    }

    // 2. 交换 dim0 和 dim1
    int tmp = idx_nd[dim0];
    idx_nd[dim0] = idx_nd[dim1];
    idx_nd[dim1] = tmp;

    // 3. 映射回新的 flat index（输出）
    int out_idx = compute_flat_index(idx_nd, shape, ndim);

    // 4. 写入输出
    output[out_idx] = input[idx];
}

void matmul(int srcX, int srcY, int dstX, int dstY, int block_size){
    int64_t batch_size = 0;
    int64_t M = 0;
    int64_t N = 0;
    int64_t K = 0;

    int64_t* send_size_d;
    int64_t send_size[4];
    cudaMalloc((void**)&send_size_d, 4*sizeof(int64_t));

    receiveMessage( srcX, srcY, dstX, dstY, send_size_d, 4*sizeof(int64_t));
    cudaMemcpy(send_size, send_size_d, 4*sizeof(int64_t), cudaMemcpyDeviceToHost);
    batch_size = send_size[0];
    M = send_size[1];
    K = send_size[2];
    N = send_size[3];

    std::cout << "batch_size: " << batch_size << std::endl;
    std::cout << "M: " << M << std::endl;
    std::cout << "K: " << K << std::endl;
    std::cout << "N: " << N << std::endl;
    

    int64_t a_size = batch_size * M * K * sizeof(float);
    int64_t b_size = batch_size * K * N * sizeof(float);
    int64_t c_size = batch_size * M * N * sizeof(float);

    std::cout << "a_size: " << a_size << std::endl;
    std::cout << "b_size: " << b_size << std::endl;
    std::cout << "c_size: " << c_size << std::endl;

    float* d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, a_size);
    cudaMalloc((void**)&d_B, b_size);
    cudaMalloc((void**)&d_C, c_size);

    // float* h_A = (float*)malloc(a_size);
    // float* h_B = (float*)malloc(b_size);
    // float* h_C = (float*)malloc(c_size);

    // for(int i = 0; i < batch_size; i++){
    //     for(int j = 0; j < M; j++){
    //         for(int k = 0; k < K; k++){
    //             h_A[i * M * K + j * K + k] = 1.0f;
    //         }
    //     }
    // }
    
    // for(int i = 0; i < batch_size; i++){
    //     for(int j = 0; j < K; j++){
    //         for(int k = 0; k < N; k++){
    //             h_B[i * K * N + j * N + k] = 1.0f;
    //         }
    //     }
    // }

    // cudaMemcpy(d_A, h_A, a_size, cudaMemcpyHostToDevice);
    // cudaMemcpy(d_B, h_B, b_size, cudaMemcpyHostToDevice);
    receiveMessage(srcX, srcY, dstX, dstY, d_A, a_size);
    receiveMessage(srcX, srcY, dstX, dstY, d_B, b_size);

    dim3 block(block_size, block_size, block_size);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y, (batch_size + block.z - 1) / block.z);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K, batch_size);
    // cudaMemcpy(h_C, d_C, c_size, cudaMemcpyDeviceToHost);
    // std::cout << "h_C: " << h_C[c_size - 1] << std::endl;

    sendMessage(dstX, dstY, srcX, srcY, d_C, c_size);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(send_size_d);

}

int main(int argc, char* argv[]){
    int srcX = atoi(argv[1]);
    int srcY = atoi(argv[2]);
    int dstX = atoi(argv[3]);
    int dstY = atoi(argv[4]);
    int block_size = atoi(argv[5]);

    bool* is_end = (bool*)malloc(sizeof(bool));
    cudaMalloc((void**)&is_end, sizeof(bool));
    while(true){
        receiveMessage(srcX, srcY, dstX, dstY, is_end, sizeof(bool));
        bool is_end_host = false;
        cudaMemcpy(&is_end_host, is_end, sizeof(bool), cudaMemcpyDeviceToHost);
        if(is_end_host){
            break;
        }
        matmul(srcX, srcY, dstX, dstY, block_size);
    }
    // cudaMemcpy(C, d_C, batch_size * M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
}