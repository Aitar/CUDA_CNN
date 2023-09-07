#ifndef MYNN_UTILS_H
#define MYNN_UTILS_H


#include <cublas_v2.h>
#include <cudnn_v8.h>
#include <memory>
#include <sstream>
#include <stdexcept>

#include "Tensor.h"


#define CUDACHECK(op) cudaCheck((op), #op, __FILE__, __LINE__)
#define CUBLASCHECK(op) cublasCheck((op), #op, __FILE__, __LINE__)
#define CUDNNCHECK(op) cudnnCheck((op), #op, __FILE__, __LINE__)
#define NIL(ptr) checkNullptr((ptr), __FILE__, __LINE__)

#define INFO(msg) logger((msg), 0)
#define WARN(msg) logger((msg), 1)
#define ERR(msg) logger((msg), 2)

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

static int INFO_RANK = 0;

void logger(std::basic_string<char, std::char_traits<char>, std::allocator<char>> msg, int type) {
    if (type < INFO_RANK) return;
    switch (type) {
        case 0: // info
            std::cout << "\033[32m[INFO] " << msg << "\033[0m" << std::endl;
            break;

        case 1: // warning
            std::cout << "\033[33m[WARNING] " << msg << "\033[0m" << std::endl;
            break;

        case 2: // error
            std::cout << "\033[31m[ERROR] " << msg << "\033[0m" << std::endl;
            break;

        default:
            std::cout << msg << std::endl;
    }
}

void cudaCheck(cudaError_t code, const char *op, const char *file, int line) {
    if (code != cudaSuccess) {
        const char *err_name = cudaGetErrorName(code);
        const char *err_message = cudaGetErrorString(code);
        printf("cuda runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name,
               err_message);
        throw std::runtime_error("cuda_error");
    }
}

void cublasCheck(cublasStatus_t code, const char *op, const char *file, int line) {
    if (code != CUBLAS_STATUS_SUCCESS) {
        const char *err_name = cublasGetErrorEnum(code);
        printf("cublas runtime error %s:%d  %s failed. \n  code = %s\n", file, line, op, err_name);
        throw std::runtime_error("cublas_error");
    }
}

void cudnnCheck(cudnnStatus_t code, const char *op, const char *file, int line) {
    if (code == CUDNN_STATUS_SUCCESS)
        return;
    std::stringstream ss;
    const char *err_name = cudnnGetErrorString(code);
    ss << "cudnn runtime error " << file << ":" << line << " " << op << " failed. \n  code = " << err_name <<"\n";
    ERR(ss.str());
    throw std::runtime_error("cudnn_error");
}

template <typename T>
T checkNullptr(T ptr, const char *file, int line) {
    if (ptr == nullptr) {
        std::stringstream ss;
        ss << "Null printer got at file " << file << ":" << line << ".";
        ERR(ss.str());
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
