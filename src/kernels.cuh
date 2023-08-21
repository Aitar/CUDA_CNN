# ifndef MYNN_KERNELS_H
# define MYNN_KERNELS_H

# include <cuda_runtime.h>
# include "utils.cuh"
# include "Tensor.h"


static const int IPT = 4;
static const int IPT_SHIFE = 2;
static const int BLOCKSIZE = 512;
static const int BLOCKSHIFE = 9;
static const int BY = 32;
static const int BX = 16;
static const int YSHIFE = 5;
static const int XSHIFE = 4;


namespace cuDL {
    __global__ void fastCopyKernel(float *src, float *dst, int dup, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float value = src[idx];   // number of threads equal to length of src
            uint offset = idx * dup;
            for (uint i = 0; i < dup; ++i)
                dst[offset + i] = value;
        }
    }


    void fastCopy(float *src, float *dst, int n, int dup = 1, cudaStream_t stream=0) {
        int blockSize = BLOCKSIZE;
        int gridSize = (n >> BLOCKSHIFE) + 1;
        fastCopyKernel<<<gridSize, blockSize, 0, stream>>>(src, dst, dup, n);
        cudaStreamSynchronize(stream);
    }

    __global__ void dSoftmaxKernel(float* dx, const float* y, float n, int w, int label) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        float tmp;
        if (idx < w) {
            tmp = idx == label ? y[idx] - 1 : y[idx];
            tmp /= n;
            dx[idx] = tmp;
        }
    }

    void dSoftmax(float* dx, const float* y, int n, int w, int label, cudaStream_t stream=0) {
        int blockSize = BLOCKSIZE;
        int gridSize = (n >> BLOCKSHIFE) + 1;
        dSoftmaxKernel<<<gridSize, blockSize, 0, stream>>>(dx, y, (float)n, w, label);
        cudaStreamSynchronize(stream);
    }
    
    
    __global__ void reduceSumExpKernel(float* x, int n) {
        uint nThread = blockIdx.x * blockDim.x;
        uint idx = nThread + threadIdx.x;
        float input[IPT] = {0.f};
        float tmp = 0.f;
        extern __shared__ float smem[];        // size == nThread

        for (uint i = idx; i < n; i += nThread * IPT) {
            for (uint step = 0; step < IPT; ++step) {
                if (i + step < n) {
                    tmp = x[i + step];
                    input[step] += exp(tmp);
                }
            }
        }
        for (uint step = 1; step < IPT; ++step)
            input[0] += input[step];

        if (threadIdx.x == 0)
            smem[threadIdx.x] = input[0];
        __syncthreads();

        for (uint stride = nThread >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }

        if (threadIdx.x == 0)
            x[blockIdx.x] = smem[0];
    }
    
    float reduceSumExpVector(float* x, int n, cudaStream_t stream=0) {
        float* copy = nullptr;
        float out = 0.f;
        cudaMallocAsync(&copy, sizeof(float) * n, stream);
        cudaMemcpyAsync(copy, x, sizeof(float) * n, cudaMemcpyDeviceToDevice, stream);

        int blockSize = BLOCKSIZE;
        int gridSize;
        int curSize = n;

        while (curSize >= blockSize) {
            gridSize = curSize / blockSize + 1;
            reduceSumExpKernel<<<gridSize, blockSize, blockSize, stream>>>(copy, curSize);
            curSize = gridSize;
        }

        blockSize = (curSize / IPT) + 1;
        reduceSumExpKernel<<<1, blockSize, blockSize, stream>>>(copy, curSize);

        cudaMemcpyAsync(&out, copy, sizeof(float), cudaMemcpyDeviceToHost, stream);

        cudaStreamSynchronize(stream);

        return out;
    }

    // y = alpha * x + beta * y
    __global__ void vectorSumKernel(float alpha, const float* x, float beta, float* y, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx << IPT_SHIFE < n) {
            float xTmp[IPT];
            float yTmp[IPT];

            for (int step = 0; step < IPT; ++step) {
                yTmp[step] = y[(idx << IPT_SHIFE) + step];
                xTmp[step] = x[(idx << IPT_SHIFE) + step];
            }

            for (int step = 0; step < IPT; ++step) {
                yTmp[step] = alpha * xTmp[step] + beta * yTmp[step];
            }

            for (int step = 0; step < IPT; ++step) {
                y[(idx << IPT_SHIFE) + step] = yTmp[step] > 0 ? yTmp[step] : 0.f;
            }
        }
    }

    void vectorSum(float alpha, const float* x, float beta, float* y, int n, cudaStream_t stream=0) {
        int blockSize = BLOCKSIZE;
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        vectorSumKernel<<<gridSize, blockSize, 0, stream>>>(alpha, x, beta, y, n);
    }
}

#endif