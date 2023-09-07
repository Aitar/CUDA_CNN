# ifndef MYNN_KERNELS_H
# define MYNN_KERNELS_H

# include <cuda_runtime.h>
# include "utils.cuh"
# include "Tensor.h"


static const int IPT = 4;
static const int IPT_SHIFE = 2;
static const int BLOCKSIZE = 512;
static const int BLOCKSHIFE = 9;


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

    __global__ void vectorSumKernel(float* x, int n, float alpha, float bias, float p) {
        uint nThread = gridDim.x * blockDim.x;
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        extern __shared__ float smem[];
        float input[IPT] = {0.f};

        // 这里很容易错，记住 i = idx * IPT
        for (uint i = idx * IPT; i < n; i += nThread * IPT) {
            for (int step = 0; step < IPT; ++step)
                input[step] += i + step < n ? pow(x[i + step] + bias, p) : 0.f;
        }

        for (uint step = 1; step < IPT; ++step)
            input[0] += input[step];
        smem[threadIdx.x] = input[0];
        __syncthreads();

        for (uint stride = blockDim.x / 2; stride >= 1; stride /= 2) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }

        if (gridDim.x == 1)
            smem[0] *= alpha;

        if (threadIdx.x == 0)
            x[blockIdx.x] = smem[0];
    }

    void vectorSum(float* x, float* res, int n, float alpha=1.f, float bias=0.f, float p=1.f, cudaStream_t stream=nullptr) {
        float* copy;
        CUDACHECK(cudaMalloc(&copy, sizeof(float) * n));
        CUDACHECK(cudaMemcpy(copy, x, sizeof(float) * n, cudaMemcpyDeviceToDevice));

        int blockSize = 256;
        int gridSize;
        bool first = true;
        while (n > 1) {
            gridSize = n / (blockSize * IPT) + 1;
            if (first) vectorSumKernel<<<gridSize, blockSize, blockSize, stream>>>(copy, n, alpha, bias, p);
            else vectorSumKernel<<<gridSize, blockSize, blockSize, stream>>>(copy, n, alpha, 0.f, 1.f);
            cudaStreamSynchronize(stream);
            first = false;
            n = gridSize;
        }
        cudaMemcpy(res, copy, sizeof(float), cudaMemcpyDeviceToHost);
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
        cudaMalloc(&copy, sizeof(float) * n);
        cudaMemcpy(copy, x, sizeof(float) * n, cudaMemcpyDeviceToDevice);

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

        cudaMemcpy(&out, copy, sizeof(float), cudaMemcpyDeviceToHost);

        cudaStreamSynchronize(stream);

        return out;
    }


    __global__ void vectorAddKernel(float alpha, const float* x, float beta, float* y, float* z, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint offset = idx << IPT_SHIFE;
        if (offset < n) {
            float xTmp[IPT];
            float yTmp[IPT];

            for (int step = 0; step < IPT; ++step) {
                yTmp[step] = y[offset + step];
                xTmp[step] = x[offset + step];
                yTmp[step] = alpha * xTmp[step] + beta * yTmp[step];
            }

            for (int step = 0; step < IPT; ++step) {
                z[offset + step] = yTmp[step];
            }
        }
    }

    // z = alpha * x + beta * y
    void vectorAdd(float alpha, const float* x, float beta, float* y, float* z, int n, cudaStream_t stream=0) {
        int blockSize = BLOCKSIZE;
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        vectorAddKernel<<<gridSize, blockSize, 0, stream>>>(alpha, x, beta, y, z, n);
    }

    // x = alpha * x + beta
    __global__ void vectorValueAddKernel(float alpha, float* x, float value, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        float input[IPT << 1] = {0.f};
        uint offset = idx * (IPT << 1);

        for (uint step = 0; step < IPT << 1; ++step) {
            if (offset + step < n) {
                input[step] = x[offset + step];
                input[step] = alpha * input[step] + value;
                x[offset + step] = input[step];
            }
        }
    }

    void vectorValueAdd(float alpha, float* x, int n, float value, cudaStream_t stream=nullptr) {
        int blockSize = BLOCKSIZE;
        int gridSize = n / (blockSize * (IPT << 1)) + 1;
        vectorValueAddKernel<<<gridSize, blockSize, 0, stream>>>(alpha, x, value, n);
    }

    __global__ void hadamardPorductKernel(float alpha, int n, const float* A, const float* B, float *C) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint offset = idx * IPT;
        if (offset < n) {
            float a[IPT] = {0.f};
            float b[IPT] = {0.f};

            for (int i = 0; i < IPT && offset + i < n; ++i) {
                a[i] = A[offset + i];
                b[i] = B[offset + i];
                a[i] *= b[i];
                a[i] *= alpha;
                C[offset + i] = a[i];
            }
        }
    }

    void hadamardPorduct(float alpha, int n, const float* A, const float* B, float* C, cudaStream_t stream=nullptr) {
        int blockSize = std::min(BLOCKSIZE, (n >> IPT_SHIFE) + 1);
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        hadamardPorductKernel<<<gridSize, blockSize, 0, stream>>>(alpha, n, A, B, C);
    }

    __global__ void expKernel(const float* x, float* y, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint offset = idx * IPT;
        if (offset < n) {
            float a[IPT] = {0.f};
            for (int i = 0; i < IPT && offset + i < n; ++i) {
                a[i] = x[offset + i];
                a[i] = std::exp(a[i]);
                y[offset + i] = a[i];
            }
        }
    }

    void exp(float* x, float* y, int n, cudaStream_t stream=nullptr) {
        int blockSize = std::min(BLOCKSIZE, (n >> IPT_SHIFE) + 1);
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        expKernel<<<gridSize, blockSize, 0, stream>>>(x, y, n);
    }

    __global__ void pullToKernel(float* x, float minValue, float maxValue, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint offset = idx * IPT;
        if (offset < n) {
            float a[IPT] = {0.f};
            for (int i = 0; i < IPT && offset + i < n; ++i) {
                a[i] = x[offset + i];
                if (a[i] > 0) {
                    a[i] = a[i] < minValue ? minValue : a[i];
                    a[i] = a[i] > maxValue ? maxValue : a[i];
                } else {
                    minValue = -minValue;
                    maxValue = -maxValue;
                    a[i] = a[i] < maxValue ? maxValue : a[i];
                    a[i] = a[i] > minValue ? minValue : a[i];
                }
                x[offset + i] = a[i];
            }
        }
    }

    void pullTo(float* x, float minValue, float maxValue, int n, cudaStream_t stream=nullptr) {
        int blockSize = std::min(BLOCKSIZE, (n >> IPT_SHIFE) + 1);
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        pullToKernel<<<gridSize, blockSize, 0, stream>>>(x, minValue, maxValue, n);
    }

    __global__ void powKernel(const float* x, float* y, float p, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint offset = idx * IPT;
        if (offset < n) {
            float a[IPT] = {0.f};
            for (int i = 0; i < IPT && offset + i < n; ++i) {
                a[i] = x[offset + i];
                if (p < 0) a[i] += 1e-9;
                a[i] = pow(a[i], p);
                y[offset + i] = a[i];
            }
        }
    }

    void pow(float* x, float* y, float p, int n, cudaStream_t stream=nullptr) {
        int blockSize = std::min(BLOCKSIZE, (n >> IPT_SHIFE) + 1);
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        powKernel<<<gridSize, blockSize, 0, stream>>>(x, y, p, n);
    }

    __global__ void logKernel(const float* x, float* y, int n) {
        uint idx = blockIdx.x * blockDim.x + threadIdx.x;
        uint offset = idx * IPT;
        if (offset < n) {
            float a[IPT] = {0.f};
            for (int i = 0; i < IPT && offset + i < n; ++i) {
                a[i] = x[offset + i];
                a[i] = std::log(a[i]);
                y[offset + i] = a[i];
            }
        }
    }

    void log(float* x, float* y, int n, cudaStream_t stream=nullptr) {
        int blockSize = std::min(BLOCKSIZE, (n >> IPT_SHIFE) + 1);
        int gridSize = n / (blockSize << IPT_SHIFE) + 1;
        logKernel<<<gridSize, blockSize, 0, stream>>>(x, y, n);
    }


    void norm(const tensorSp& x) {
        auto mean = std::make_shared<Tensor>(1);
        auto var = std::make_shared<Tensor>(1);

        vectorSum(x->gpu(), mean->gpu(), x->size(), 1.f/(float) x->size());
        vectorSum(x->gpu(), var->gpu(), x->size(), 1.f/(float) x->size(), -mean->getItem(0), 2.f);
        float std = std::pow(var->getItem(0) + 1e-5, -0.5);
        mean->cpu()[0] *= std;
        vectorValueAdd(std, x->gpu(), x->size(), -mean->getItem(0));
    }
}

#endif