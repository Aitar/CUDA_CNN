#ifndef MYNN_UTILS_H
#define MYNN_UTILS_H


#include <cublas_v2.h>
#include <cudnn_v8.h>
#include <memory>
#include "Tensor.h"


#define CUDACHECK(op) cudaCheck((op), #op, __FILE__, __LINE__)
#define CUBLASCHECK(op) cublasCheck((op), #op, __FILE__, __LINE__)
#define CUDNNCHECK(op) cudnnCheck((op), #op, __FILE__, __LINE__)

static const char* cublasGetErrorEnum(cublasStatus_t error) {
    switch (error) {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "CUBLAS_STATUS_NOT_SUPPORTED";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "CUBLAS_STATUS_LICENSE_ERROR";

        default:
            return "None";
    }
}

bool cudaCheck(cudaError_t code, const char *op, const char *file, int line) {
    if (code != cudaSuccess) {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("cuda runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name,
               err_message);
        return false;
    }
    return true;
}

bool cublasCheck(cublasStatus_t code, const char *op, const char *file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        const char *err_name = cublasGetErrorEnum(code);
        printf("cublas runtime error %s:%d  %s failed. \n  code = %s\n", file, line, op, err_name);
        return false;
    }
    return true;
}

bool cudnnCheck(cudnnStatus_t code, const char *op, const char *file, int line) {
    if (code == CUDNN_STATUS_SUCCESS)
        return true;
    const char *err_name = cudnnGetErrorString(code);
    printf("cudnn runtime error %s:%d  %s failed. \n  code = %s\n", file, line, op, err_name);
    return false;
}

template <typename T>
std::shared_ptr<T> ckNullptr(std::shared_ptr<T> ptr) {
    if (ptr == nullptr) {
        std::cout << "[Error] Nullptr got." << std::endl;
        exit(1);
    }
    return ptr;
}

namespace cuDL {
    class CudaContext {
    public:
        cublasHandle_t cublas_{};
        cudnnHandle_t cudnn_{};
        const float one = 1.f;
        const float zero = 0.f;
        const float minusOne = -1.f;

        CudaContext() {
            cublasCreate(&cublas_);
            CUDACHECK(cudaGetLastError());
            CUDNNCHECK(cudnnCreate(&cudnn_));
        }

        ~CudaContext() {
            CUBLASCHECK(cublasDestroy(cublas_));
            CUDNNCHECK(cudnnDestroy(cudnn_));
        }
    };

    int toInt(const uint8_t *ptr) {
        return ((ptr[0] & 0xFF) << 24 | (ptr[1] & 0xFF) << 16 |
                (ptr[2] & 0xFF) << 8 | (ptr[3] & 0xFF) << 0);
    }
}

#endif //MYNN_UTILS_H
