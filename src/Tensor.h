//
// Created by Aitar Hwan on 2023/3/11.
//

#ifndef MYNN_TENSOR_H
#define MYNN_TENSOR_H

# include <array>
# include <string>
# include <iostream>
# include <fstream>
# include <random>
# include <memory>

# include "utils.cuh"
# include "kernels.cuh"

namespace cuDL {
    typedef enum {CPU, GPU} Device;

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

        void tensorInit(int n, int c, int h, int w, float* arr=nullptr) {
            n_ = n;
            c_ = c;
            h_ = h;
            w_ = w;
            cpu_ = new float[n * c * w * h];
            if (arr != nullptr)
                arrayInit(arr);
            CUDACHECK(cudaMalloc(&gpu_, sizeof(float) * n * c * h * w));
            CUDNNCHECK(cudnnCreateTensorDescriptor(&desc_));
            CUDNNCHECK(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n, c, h, w));
        }

    public:
        cudnnTensorDescriptor_t desc_{};

        Tensor(int n, int c, int h, int w) {
            tensorInit(n, c, h, w);
        }

        Tensor(std::shared_ptr<Tensor> tensor, int dup=1) {
            tensorInit(tensor->n() * dup, tensor->c(), tensor->h(), tensor->w());
            for (int i = 0; i < dup; ++i)
                cudaMemcpyAsync(gpu() + i * tensor->size(), tensor->gpu(), tensor->memSize(), cudaMemcpyDeviceToDevice);
        }

        Tensor(int n, int c, int h, int w, float* arr) {
            tensorInit(n, c, h, w, arr);
        }

        void valueInit(float value=0.f) {
            for (int i = 0; i < size(); ++i)
                cpu()[i] = value;
        }

        void arrayInit(float* arr) {
            for (int i = 0; i < size(); ++i)
                cpu()[i] = arr[i];
        }

        void uniformInit() {
            std::random_device rd;
            std::mt19937 gen(rd());
            float range = std::sqrt(6.f / w());
            std::uniform_real_distribution<> dis(-range, range);

            for (int i = 0; i < size(); ++i)
                cpu()[i] = static_cast<float>(dis(gen));
        }

        void print() {
            for (int n = 0; n < n_; ++n) {
                for (int c = 0; c < c_; ++c) {
                    printf("\nN = %d, C = %d:\n", n, c);
                    for (int h = 0; h < h_; ++h) {
                        for (int w = 0; w < w_; ++w)
                            printf("%8.4f, ", getItem(n, c, h, w));
                        printf("\n");
                    }
                }
            }
        }

        void printMem() {
            printf("\n");
            for (int n = 0; n < size(); ++n) {
                printf("%8.4f, ", cpu()[n]);
            }
            printf("\n");
        }

        ~Tensor() {
            tensorFree();
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
                return -1;
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
        void setItem(int n, int c, int h, int w, float v) {
            cpu()[n * c_ * h_ * w_ +
                  c * h_ * w_ +
                  h * w_ +
                  w] = v;
        }
        int size() const {return n_ * c_ * h_ * w_; }
        int len() const {return c_ * h_ * w_; }
        size_t memSize() const {return size() * sizeof(float); }
        int n() const { return n_; }
        int c() const { return c_; }
        int h() const { return h_; }
        int w() const { return w_; }
    };
}

#endif //MYNN_TENSOR_H
