#ifndef MYNN_TENSOR_H
#define MYNN_TENSOR_H

# include <array>
# include <string>
# include <iostream>
# include <fstream>
# include <random>
# include <memory>
#include <utility>

# include "utils.cuh"

namespace cuDL {

__global__ void setValueKernel(float* x, float value, int n) {
    uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint offset = idx * 8;
    if (offset < n) {
        for (int i = 0; i < 8 && offset + i < n; ++i)
            x[offset + i] = value;
    }
}

void setValue(float* x, float value, int n, cudaStream_t stream=nullptr) {
    int blockSize = std::min(512, (n >> 3) + 1);
    int gridSize = n / (blockSize << 3) + 1;
    setValueKernel<<<gridSize, blockSize, 0, stream>>>(x, value, n);
}

typedef enum {CPU, GPU} Device;

/**
 * 一维：size
 * 二维：n, len
 * 三维：n, c, area
 * 四维：n, c, h, w
 */
class Tensor {
private:
    float* gpu_ = nullptr;
    float* cpu_ = nullptr;
    Device device_ = CPU;
    int n_ = 1;
    int c_ = 1;
    int h_ = 1;
    int w_ = 1;

    void tensorFree() {
        if (gpu_ != nullptr) cudaFree(gpu_);
        delete [] cpu_;
        cudnnDestroyTensorDescriptor(desc_);
    }

    void tensorInit(int n, int c, int h, int w) {
        n_ = n;
        c_ = c;
        h_ = h;
        w_ = w;

        cpu_ = new float[n * c * w * h];
        CUDACHECK(cudaMalloc(&gpu_, sizeof(float) * n * c * h * w));
        CUDNNCHECK(cudnnCreateTensorDescriptor(&desc_));
        CUDNNCHECK(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
    }

public:
    cudnnTensorDescriptor_t desc_{};

    Tensor(int w, float* arr=nullptr) {
        tensorInit(1, 1, 1, w);
        if (arr != nullptr) arrayInit(arr);
        else valueInit();
    }

    Tensor(int n, int w, float* arr=nullptr) {
        tensorInit(n, 1, 1, w);
        if (arr != nullptr) arrayInit(arr);
        else valueInit();
    }

    Tensor(int n, int h, int w, float* arr=nullptr) {
        tensorInit(n, 1, h, w);
        if (arr != nullptr) arrayInit(arr);
        else valueInit();
    }

    Tensor(int n, int c, int h, int w, float* arr=nullptr) {
        tensorInit(n, c, h, w);
        if (arr != nullptr) arrayInit(arr);
        else valueInit();
    }

    Tensor(const std::shared_ptr<Tensor>& tensor, bool copy=false, int dup=1) {
        tensorInit(tensor->n(), tensor->c() * dup, tensor->h(), tensor->w());
        if (copy) {
            for (int i = 0; i < dup; ++i)
                cudaMemcpy(gpu() + i * tensor->size(),
                                tensor->gpu(),
                                tensor->memSize(),
                                cudaMemcpyDeviceToDevice);

        } else {
            valueInit();
        }
    }

    Tensor(std::vector<std::shared_ptr<Tensor>> ts, cudaStream_t stream=nullptr) {
        auto tp = NIL(ts[0]);
        int s = tp->size();
        tensorInit(tp->n() * ts.size(), tp->c(), tp->h(), tp->w());
        for (int i = 0; i < ts.size(); ++i) {
            tp = NIL(std::move(ts[i]));
            cudaMemcpy(this->gpu() + i * s, tp->gpu() ,
                            sizeof(float) * s,
                            cudaMemcpyDeviceToDevice);
        }
    }

    std::vector<std::shared_ptr<Tensor>> split(int part, cudaStream_t stream=nullptr) {
        if (n_ % part != 0) {
            ERR("Tensor N must be evenly divided by split parts.");
            exit(1);
        }
        int s = n_ / part;
        std::vector<std::shared_ptr<Tensor>> ts;
        for (int i = 0; i < part; ++i) {
            auto t = std::make_shared<Tensor>(s);
            if (device_ == GPU)
                cudaMemcpy(t->gpu(), this->gpu() + i * s,
                                sizeof(float) * s,
                                cudaMemcpyDeviceToDevice);
            else
                memcpy(t->cpu(), this->cpu() + i * s, sizeof(float) * s);
            ts.push_back(std::move(t));
        }
        return ts;
    }

    void valueInit(float value=0.f) {
        if (device_ == CPU) {
            for (int i = 0; i < size(); ++i)
                cpu()[i] = value;
        } else {
            setValue(gpu(), value, size());
        }
    }

    void arrayInit(const float* arr) {
        memcpy(cpu(), arr, sizeof(float) * size());
    }

    void uniformInit() {
        std::random_device rd;
        std::mt19937 gen(rd());
        float range = std::sqrt(1.f / w());
        std::uniform_real_distribution<> dis(-range, range);

        for (int i = 0; i < size(); ++i)
            cpu()[i] = static_cast<float>(dis(gen));
    }

    void print() {
        for (int n = 0; n < n_; ++n) {
            for (int c = 0; c < c_; ++c) {
                printf("N = %d, C = %d:\n", n, c);
                for (int h = 0; h < h_; ++h) {
                    for (int w = 0; w < w_; ++w)
                        printf("%8.4f, ", getItem(n, c, h, w));
                    printf("\n\n");
                }
            }
        }
    }

    void printMem() {
        printf("\n");
        for (int n = 0; n < size(); ++n) {
            printf("%6.4f, ", cpu()[n]);
        }
        printf("\n");
    }

    void printShape() {
        printf("\n");
        printf("(%d, %d, %d, %d)\n", n_, c_, h_, w_);
    }

    ~Tensor() {
        tensorFree();
    }

    void copy(const std::shared_ptr<Tensor>& tensor, int dup=1) {
        for (int i = 0; i < dup; ++i) {
            if (tensor->device_ == GPU) {
                cudaMemcpy(gpu() + i * tensor->size(), tensor->gpu(), tensor->memSize(), cudaMemcpyDeviceToDevice);
            } else {
                memcpy(cpu() + i * tensor->size(), tensor->cpu(), tensor->memSize());
            }
        }
    }

    float* cpu() {
        if (device_ == GPU) {
            cudaMemcpy(cpu_, gpu_, sizeof(float) * size(), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            device_ = CPU;
        }
        return cpu_;
    }

    float* gpu() {
        if (device_ == CPU) {
            cudaMemcpy(gpu_, cpu_, sizeof(float) * size(), cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();
            device_ = GPU;
        }
        return gpu_;
    }

    std::array<int, 4> shape() {
        return std::array<int, 4>({n_, c_, h_, w_});
    }

    void reset(int n, int c, int h, int w) {
        n_ = n; c_ = c; h_ = h; w_ = w;
        tensorFree();
        tensorInit(n, c, h, w);
    }

    int reshape(int c, int h, int w) {
        if (c * h * w != c_ * h_ * w_) {
            printf("[ERROR] Shape mismatch, trying %d, %d, %d to %d, %d, %d\n", c_, h_, w_, c, h, w);
            exit(-1);
        }
        c_ = c; h_ = h; w_ = w;
        return 0;
    }
    float getItem(int n, int c, int h, int w) {
        return cpu()[n * c_ * h_ * w_ +
                     c * h_ * w_ +
                     h * w_ +
                     w];
    }

    float getItem(int n, int h, int w) {
        return cpu()[n * len() + h * w_ + w];
    }

    float getItem(int n, int w) {
        return cpu()[n * len() + w];
    }

    float getItem(int w) {
        return cpu()[w];
    }

    void setItem(int n, int c, int h, int w, float v) {
        cpu()[n * c_ * h_ * w_ +
              c * h_ * w_ +
              h * w_ +
              w] = v;
    }

    void setItem(int n, int h, int w, float v) {
        cpu()[n * len() + h * w_ + w] = v;
    }

    void setItem(int n, int w, float v) {
        cpu()[n * len() + w] = v;
    }

    void setItem(int n, float v) {
        cpu()[n] = v;
    }

    Device device() { return device_; }
    [[nodiscard]] int size() const {return n_ * c_ * h_ * w_; }
    [[nodiscard]] int len() const {return c_ * h_ * w_; }
    [[nodiscard]] int area() const {return h_ * w_; }
    [[nodiscard]] size_t memSize() const {return size() * sizeof(float); }
    [[nodiscard]] int n() const { return n_; }
    [[nodiscard]] int c() const { return c_; }
    [[nodiscard]] int h() const { return h_; }
    [[nodiscard]] int w() const { return w_; }
};

typedef std::shared_ptr<Tensor> tensorSp;
typedef std::vector<std::shared_ptr<Tensor>> tensors;

tensorSp onehot(const tensorSp& t, int nClass) {
    auto res = std::make_shared<Tensor>(t->size(), nClass);
    for (int i = 0; i < t->n(); ++i) {
        res->setItem(i, (int)t->getItem(i), 1.f);
    }
    return res;
}

}

#endif //MYNN_TENSOR_H
