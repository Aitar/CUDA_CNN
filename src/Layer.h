//
// Created by Aitar Hwan on 2023/3/11.
//

#ifndef MYNN_LAYER_H
#define MYNN_LAYER_H


# include <cmath>
# include <memory>
# include <string>
# include <utility>
# include <vector>
# include <map>
# include <random>
# include <algorithm>


# include "Tensor.h"
# include "utils.cuh"

namespace cuDL {
    static int globalID = 0;

    class Layer {
    protected:
        std::string name_;
        std::shared_ptr <Tensor> outputs_ = nullptr;          // 需要根据batchSize初始化
        std::shared_ptr <Tensor> inputs_ = nullptr;
        std::shared_ptr <Tensor> inputGrads_ = nullptr;
        std::shared_ptr <Tensor> outputsGrads_ = nullptr;
        std::shared_ptr <CudaContext> cudaCtx_ = nullptr;
        std::map <std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> parmas_;
        cudaStream_t stream_{};

        int batchSize_ = 0;
        bool isInit = false;
        bool hasParams_ = false;

        friend class Module;

    public:
        virtual std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> inputs) = 0;

        virtual std::shared_ptr <Tensor> backward(std::shared_ptr <Tensor> grads) = 0;

        virtual void init(std::shared_ptr <Tensor> inputs) {
            isInit = true;
            inputs_ = std::move(inputs);
            batchSize_ = inputs_->n();
            inputGrads_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), inputs_->h(), inputs_->w());
            inputGrads_->valueInit();
        };

        virtual void updateParams(float lr) {
            if (!hasParams_) {
                printf("[Warning] Try to update parameters at layer without parameters.\n");
                return;
            }
            for (auto &param: parmas_)
                vectorSum(lr, param.second->gpu(), 1.f, param.first->gpu(), param.first->size());

            cudaDeviceSynchronize();
        }

        std::string getName() { return name_; };

        void setCudaContext(std::shared_ptr <CudaContext> cudaCtx) {
            cudaCtx_ = std::move(cudaCtx);
        }

        virtual void print() {}
    };

    class Linear : public Layer {
    public:
        std::shared_ptr <Tensor> weight_ = nullptr;  // (1, 1, inSize_, outSize_)
        std::shared_ptr <Tensor> bias_ = nullptr;
        std::shared_ptr <Tensor> wGrad_ = nullptr;
        std::shared_ptr <Tensor> bGrad_ = nullptr;
        int inSize_ = 0;
        int outSize_ = 0;

        void init(std::shared_ptr <Tensor> inputs) override {
            if (isInit) return;
//            printf("\nLinear init...");
            Layer::init(inputs);

            weight_ = std::make_shared<Tensor>(1, 1, inSize_, outSize_);
            bias_ = std::make_shared<Tensor>(1, 1, 1, outSize_);
            wGrad_ = std::make_shared<Tensor>(1, 1, inSize_, outSize_);
            bGrad_ = std::make_shared<Tensor>(1, 1, 1, outSize_);

            weight_->uniformInit();
            bias_->valueInit();
            wGrad_->valueInit();
            bGrad_->valueInit();
            outputs_ = std::make_shared<Tensor>(bias_, batchSize_);

            parmas_[weight_] = wGrad_;
            parmas_[bias_] = bGrad_;

//            printf("success!\n");
        }

        Linear(int inSize, int outSize) {
            name_ = "Linear_" + std::to_string(globalID++);
            inSize_ = inSize;
            outSize_ = outSize;
            hasParams_ = true;
        }

        ~Linear() = default;

        std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> inputs) override {
            if (inputs->len() != inSize_)
                printf("\nShape dismatch, expect %d but got %d\n", inSize_, inputs->size());

            init(inputs);
            inputs_->reshape(1, 1, inputs_->c() * inputs_->h() * inputs_->w());
//            printf("\nLinear forward...");

            // YT = WT * XT + BT
            // WT: O * I, XT: I * N, YT = O * N
            CUBLASCHECK(cublasSgemm_v2(cudaCtx_->cublas_,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       outSize_, batchSize_, inSize_,
                                       &cudaCtx_->one,
                                       weight_->gpu(), outSize_,
                                       inputs_->gpu(), inSize_,
                                       &cudaCtx_->one,
                                       outputs_->gpu(), outSize_));

//            printf("success!\n");
            return outputs_;
        }

        std::shared_ptr <Tensor> backward(std::shared_ptr <Tensor> grads) override {
//            printf("\nLinear backward...");

            // db = dy
            outputsGrads_ = std::move(grads);
            bGrad_ = outputsGrads_;

            // dw = x * (dy)^T
            CUBLASCHECK(cublasSger_v2(cudaCtx_->cublas_,
                                      outSize_, inSize_,
                                      &cudaCtx_->one,
                                      outputsGrads_->gpu(), 1,      // dy: 1, 1, 1, outSize_
                                      inputs_->gpu(), 1,            // x:  1, 1, 1, inSize_
                                      wGrad_->gpu(), outSize_));     // dw: 1, 1, outSize_, inSize_

            // dx = w * (dy)^T
            // y = alpha * A @ x + beta * y
            CUBLASCHECK(cublasSgemv(cudaCtx_->cublas_,
                                    CUBLAS_OP_T,
                                    outSize_, inSize_,
                                    &cudaCtx_->one,
                                    weight_->gpu(), outSize_,
                                    outputsGrads_->gpu(), 1,    // dy: (1, 1, 1, outSize_)
                                    &cudaCtx_->zero,
                                    inputGrads_->gpu(), 1));    // dx: (1, 1, 1, inSize_)
            cudaDeviceSynchronize();
//            printf("success!\n");
            return inputGrads_;
        }

        void print() override {
            cudaDeviceSynchronize();

            printf("\n%s\n", name_.c_str());

            printf("weight:");
            if (weight_ != nullptr) weight_->print();
            else printf("weight not init yet\n");

            printf("\nbias:");
            if (bias_ != nullptr) bias_->print();
            else printf("bias not init yet\n");

            printf("\nweight grad:");
            if (wGrad_ != nullptr) wGrad_->print();
            else printf("wGrad_ not init yet\n");

            printf("\ninput grad:");
            if (inputGrads_ != nullptr) inputGrads_->print();
            else printf("inputGrads_ not init yet\n");

            printf("\nbias grad:");
            if (bGrad_ != nullptr) bGrad_->print();
            else printf("bGrads_ not init yet\n");
        }
    };


    class Conv2D : public Layer {
    private:
        std::shared_ptr <Tensor> kernel_ = nullptr;
        std::shared_ptr <Tensor> bias_ = nullptr;
        std::shared_ptr <Tensor> kernelGrads_ = nullptr;
        std::shared_ptr <Tensor> biasGrads_ = nullptr;

        cudnnFilterDescriptor_t filterDesc_{};
        cudnnConvolutionDescriptor_t convDesc_{};

        cudnnConvolutionFwdAlgo_t fwdAlgo_;
        cudnnConvolutionBwdDataAlgo_t bwdDataAlgo_;
        cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo_;

        void **workspace_{};
        size_t workspaceSize_ = 0;

    public:
        std::string name_;
        int inChannels_;
        int outChannels_;
        int kernelSize_;
        int stride_;
        int padding_;
        int dilation_;
        int inputH_{};
        int inputW_{};
        int outputH_{};
        int outputW_{};
        float paddingFill_;


        Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0,
               float paddingFill = 0.f, int dilation = 1) :
                inChannels_(inChannels),
                outChannels_(outChannels),
                kernelSize_(kernelSize),
                stride_(stride),
                padding_(padding),
                paddingFill_(paddingFill),
                dilation_(dilation) {

            name_ = "Conv2D_" + std::to_string(globalID);
            hasParams_ = true;
            CUDNNCHECK(cudnnCreateFilterDescriptor(&filterDesc_));
            CUDNNCHECK(cudnnCreateConvolutionDescriptor(&convDesc_));
            CUDNNCHECK(cudnnSetConvolution2dDescriptor(convDesc_,
                                                       padding_,
                                                       padding_,
                                                       stride_,
                                                       stride_,
                                                       dilation_,
                                                       dilation_,
                                                       CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
            CUDNNCHECK(cudnnSetConvolutionMathType(convDesc_, CUDNN_DEFAULT_MATH));
        }

        ~Conv2D() {
            cudnnDestroyFilterDescriptor(filterDesc_);
            cudnnDestroyConvolutionDescriptor(convDesc_);
        }

        void setWorkspace() {
//            std::cout << "\nConv set workspace...";

            /** MaxCount -> v7(find best) -> WorkSpace
             *  遍历一遍找到最合适的算法，并设置workspace大小
             */
            int maxAlgoCnt;
            int algoCnt = 0;
            size_t tempSize = 0;

            std::vector <cudnnConvolutionFwdAlgoPerf_t> fwdPerfs(CUDNN_CONVOLUTION_FWD_ALGO_COUNT);
            std::vector <cudnnConvolutionBwdFilterAlgoPerf_t> bwdFilterPerfs(CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT);
            std::vector <cudnnConvolutionBwdDataAlgoPerf_t> bwdDataPerfs(CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT);

            // forward 找到最合适的卷积算法，并设置workspace
            CUDNNCHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(cudaCtx_->cudnn_, &maxAlgoCnt));
            CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm_v7(cudaCtx_->cudnn_,
                                                              inputs_->desc_,
                                                              filterDesc_,
                                                              convDesc_,
                                                              outputs_->desc_,
                                                              maxAlgoCnt,
                                                              &algoCnt,
                                                              &fwdPerfs[0]));
            fwdAlgo_ = fwdPerfs[0].algo;
            CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(cudaCtx_->cudnn_,
                                                               inputs_->desc_,
                                                               filterDesc_,
                                                               convDesc_,
                                                               outputs_->desc_,
                                                               fwdAlgo_, &tempSize));
            workspaceSize_ = std::max(workspaceSize_, tempSize);

            // bias 找到最合适的反向传播计算算法，
            CUDNNCHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudaCtx_->cudnn_, &maxAlgoCnt));
            CUDNNCHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudaCtx_->cudnn_,
                                                                     inputs_->desc_,
                                                                     outputs_->desc_,
                                                                     convDesc_,
                                                                     filterDesc_,
                                                                     maxAlgoCnt,
                                                                     &algoCnt,
                                                                     &bwdFilterPerfs[0]));
            bwdFilterAlgo_ = bwdFilterPerfs[0].algo;
            CUDNNCHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudaCtx_->cudnn_,
                                                                      inputs_->desc_,
                                                                      outputs_->desc_,
                                                                      convDesc_,
                                                                      filterDesc_,
                                                                      bwdFilterAlgo_, &tempSize));
            workspaceSize_ = std::max(workspaceSize_, tempSize);

            // data - bwd
            CUDNNCHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudaCtx_->cudnn_, &maxAlgoCnt));
            CUDNNCHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudaCtx_->cudnn_,
                                                                   filterDesc_,
                                                                   outputs_->desc_,
                                                                   convDesc_,
                                                                   inputs_->desc_,
                                                                   maxAlgoCnt,
                                                                   &algoCnt,
                                                                   &bwdDataPerfs[0]));
            bwdDataAlgo_ = bwdDataPerfs[0].algo;
            CUDNNCHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cudaCtx_->cudnn_,
                                                                    filterDesc_,
                                                                    outputs_->desc_,
                                                                    convDesc_,
                                                                    inputs_->desc_,
                                                                    bwdDataAlgo_, &tempSize));
            workspaceSize_ = std::max(workspaceSize_, tempSize);

            if (workspaceSize_ > 0) {
                if (workspace_ != nullptr)
                    CUDACHECK(cudaFree(workspace_));
                CUDACHECK(cudaMalloc((void **) &workspace_, workspaceSize_));
            }
//            printf("success\n");
        }

        void init(std::shared_ptr <Tensor> inputs) override {
            if (isInit) return;

//            printf("\nConv init...");

            Layer::init(inputs);
            inputH_ = inputs->h();
            inputW_ = inputs->w();
            outputH_ = (inputH_ + 2 * padding_ - kernelSize_) / stride_ + 1;
            outputW_ = (inputW_ + 2 * padding_ - kernelSize_) / stride_ + 1;

            kernel_ = std::make_shared<Tensor>(inChannels_, outChannels_, kernelSize_, kernelSize_);
            outputs_ = std::make_shared<Tensor>(batchSize_, outChannels_, outputH_, outputW_);
            kernelGrads_ = std::make_shared<Tensor>(inChannels_, outChannels_, kernelSize_, kernelSize_);
            bias_ = std::make_shared<Tensor>(1, outChannels_, 1, 1);
            biasGrads_ = std::make_shared<Tensor>(1, outChannels_, 1, 1);

            kernel_->uniformInit();
            outputs_->valueInit();
            kernelGrads_->valueInit();
            bias_->valueInit();
            biasGrads_->valueInit();

            parmas_[kernel_] = kernelGrads_;
            parmas_[bias_] = biasGrads_;

            CUDNNCHECK(cudnnSetFilter4dDescriptor(filterDesc_,
                                                  CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                                  outChannels_, inChannels_, kernelSize_, kernelSize_));

//            printf("success\n");
            setWorkspace();
        }

        void print() override {
            printf("\n%s", name_.c_str());
            printf("\nkernel:");
            if (kernel_ != nullptr) kernel_->print();
            else printf("not init yet\n");

            printf("\nbias:");
            if (bias_ != nullptr) bias_->print();
            else printf("not init yet\n");

            printf("\nkernelGrads:");
            if (kernelGrads_ != nullptr) kernelGrads_->print();
            else printf("not init yet\n");

            printf("\nbiasGrads_:");
            if (biasGrads_ != nullptr) biasGrads_->print();
            else printf("not init yet\n");
        }

        std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> inputs) override {
            init(inputs);
//            printf("\nConv forward...");
            CUDNNCHECK(cudnnConvolutionForward(cudaCtx_->cudnn_,
                                               &cudaCtx_->one, inputs_->desc_, inputs_->gpu(),
                                               filterDesc_, kernel_->gpu(),
                                               convDesc_, fwdAlgo_,
                                               workspace_, workspaceSize_,
                                               &cudaCtx_->zero, outputs_->desc_, outputs_->gpu()));

            CUDNNCHECK(cudnnAddTensor(cudaCtx_->cudnn_,
                                      &cudaCtx_->one, bias_->desc_, bias_->gpu(),
                                      &cudaCtx_->one, outputs_->desc_, outputs_->gpu()));
//            printf("success\n");
            return outputs_;
        }

        std::shared_ptr <Tensor> backward(std::shared_ptr <Tensor> grads) override {
            outputsGrads_ = std::move(grads);

            CUDNNCHECK(cudnnConvolutionBackwardBias(cudaCtx_->cudnn_,
                                                    &cudaCtx_->one, outputsGrads_->desc_, outputsGrads_->gpu(),
                                                    &cudaCtx_->zero, biasGrads_->desc_, biasGrads_->gpu()));

            CUDNNCHECK(cudnnConvolutionBackwardFilter(cudaCtx_->cudnn_,
                                                      &cudaCtx_->one, inputs_->desc_, inputs_->gpu(),
                                                      outputsGrads_->desc_, outputsGrads_->gpu(),
                                                      convDesc_, bwdFilterAlgo_,
                                                      workspace_, workspaceSize_,
                                                      &cudaCtx_->zero, filterDesc_, kernelGrads_->gpu()));

            CUDNNCHECK(cudnnConvolutionBackwardData(cudaCtx_->cudnn_,
                                                    &cudaCtx_->one, filterDesc_, kernel_->gpu(),
                                                    outputsGrads_->desc_, outputsGrads_->gpu(),
                                                    convDesc_, bwdDataAlgo_,
                                                    workspace_, workspaceSize_,
                                                    &cudaCtx_->zero, inputGrads_->desc_, inputGrads_->gpu()));
            return inputGrads_;
        }
    };

    class Pooling : public Layer {
    private:
        int kernelSize_;
        int padding_;
        int stride_;
        int inputH_{};
        int inputW_{};
        int outputH_{};
        int outputW_{};
        cudnnPoolingMode_t mode_;
        cudnnPoolingDescriptor_t poolingDesc_{};

    public:
        Pooling(int kernelSize, int padding, int stride, cudnnPoolingMode_t mode) :
                kernelSize_(kernelSize), padding_(padding), stride_(stride), mode_(mode) {

            cudnnCreatePoolingDescriptor(&poolingDesc_);
            cudnnSetPooling2dDescriptor(poolingDesc_,
                                        mode_, CUDNN_PROPAGATE_NAN,
                                        kernelSize_, kernelSize_,
                                        padding_, padding_,
                                        stride_, stride_);
        }

        ~Pooling() {
            cudnnDestroyPoolingDescriptor(poolingDesc_);
        }

        void init(std::shared_ptr <Tensor> inputs) override {
            if (isInit) return;

//            printf("\nPooling init...");
            Layer::init(inputs);
            inputH_ = inputs->h();
            inputW_ = inputs->w();
            outputH_ = (inputH_ + 2 * padding_ - kernelSize_) / stride_ + 1;
            outputW_ = (inputW_ + 2 * padding_ - kernelSize_) / stride_ + 1;
            outputs_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), outputH_, outputW_);
            outputs_->valueInit();
//            printf("Success\n");
        }

        std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> inputs) override {
            init(inputs);
//            printf("\nPooling forward...");
            CUDNNCHECK(cudnnPoolingForward(cudaCtx_->cudnn_, poolingDesc_,
                                           &cudaCtx_->one, inputs_->desc_, inputs_->gpu(),
                                           &cudaCtx_->zero, outputs_->desc_, outputs_->gpu()));
//            printf("Success\n");
            return outputs_;
        }

        std::shared_ptr <Tensor> backward(std::shared_ptr <Tensor> grads) override {
            outputsGrads_ = std::move(grads);
            CUDNNCHECK(cudnnPoolingBackward(cudaCtx_->cudnn_, poolingDesc_,
                                            &cudaCtx_->one,
                                            outputs_->desc_, outputs_->gpu(),
                                            outputsGrads_->desc_, outputsGrads_->gpu(),
                                            inputs_->desc_, inputs_->gpu(),
                                            &cudaCtx_->zero,
                                            inputGrads_->desc_, inputGrads_->gpu()));

            return inputGrads_;
        }
    };

    class MaxPooling : public Pooling {
    public:
        explicit MaxPooling(int kernelSize, int padding = 0, int stride = 1) :
                Pooling(kernelSize, padding, stride, CUDNN_POOLING_MAX) {
            name_ = "MaxPooling_" + std::to_string(globalID++);
        }
    };

    class AvgPooling : public Pooling {
    public:
        explicit AvgPooling(int kernelSize = 2, int padding = 0, int stride = 1) :
                Pooling(kernelSize, padding, stride, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
            name_ = "MaxPooling_" + std::to_string(globalID++);
        }
    };

    /**
     * softmax层
     * input shape = (batchSize, 1, 1, size)
     * output shape = (batchSize, 1, 1, size)
     */
    class Softmax : public Layer {
    public:
        int size_ = 0;
        std::shared_ptr <Tensor> loss_;

        Softmax() {
            name_ = "Softmax_" + std::to_string(globalID++);
        }

        void init(std::shared_ptr <Tensor> inputs) override {
            if (isInit) return;

            Layer::init(inputs);
            size_ = inputs->w();
            outputs_ = std::make_shared<Tensor>(batchSize_, 1, 1, size_);
            inputGrads_ = std::make_shared<Tensor>(batchSize_, 1, 1, size_);
            loss_ = std::make_shared<Tensor>(1, 1, 1, 1);
            outputs_->valueInit();
            loss_->valueInit();
        }

        std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> inputs) override {
            init(inputs);
//            std::cout << "[Info] Softmax forward...";
            CUDNNCHECK(cudnnSoftmaxForward(cudaCtx_->cudnn_,
                                           CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                           &cudaCtx_->one, inputs_->desc_, inputs_->gpu(),
                                           &cudaCtx_->zero, outputs_->desc_, outputs_->gpu()));
            printf("-------------\n");
            inputs_->print();
            outputs_->print();
            return outputs_;
        }

        float getLoss(const std::shared_ptr <Tensor> &labels) {
            float loss = 0.f;
            for (int i = 0; i < outputs_->n(); ++i) {
                dSoftmax(inputGrads_->gpu() + i * outputs_->len(),
                         outputs_->gpu() + i * outputs_->len(),
                         batchSize_,
                         size_,
                         (int) labels->cpu()[i]);
                loss -= std::log(outputs_->getItem(i, 0, 0, (int) labels->cpu()[i]) + 1e-5);
            }
            return loss / outputs_->n();
        }

//        std::shared_ptr<Tensor> forwardAndLCE(std::shared_ptr<Tensor> tensor, std::shared_ptr<Tensor> labels) {
//            float expSums[tensor->n()];
//            auto res = std::make_shared<Tensor>(1, 1, 1, 1);
//            res->valueInit();
//            for (int i = 0; i < tensor->n(); ++i)
//                expSums[i] = std::log(reduceSumExpVector(tensor->gpu() + i * tensor->len(), tensor->len()));
//            for (int i = 0; i < tensor->n(); ++i) {
//                res->cpu()[0] += expSums[i];
//                res->cpu()[0] -= tensor->cpu()[(int) labels->cpu()[i]];
//            }
//            return res;
//        }

        std::shared_ptr <Tensor> backward(std::shared_ptr <Tensor> grads) override {
            if (grads != nullptr) {
                printf("[Warning] Softmax backward only accept nullptr.\n");
                return nullptr;
            }
            return inputGrads_;
        }

        void print() {
            printf("\n%s", name_.c_str());
            printf("\ninputGrads:");
            if (inputGrads_ != nullptr) inputGrads_->print();
            else printf("not init yet\n");
        }
    };


    class Activation : public Layer {
    private:
        cudnnActivationDescriptor_t actDesc_{};
        cudnnActivationMode_t mode_;
        float coef_;

    public:
        Activation(float coef, cudnnActivationMode_t mode) {
            mode_ = mode;
            coef_ = coef;
            cudnnCreateActivationDescriptor(&actDesc_);
            cudnnSetActivationDescriptor(actDesc_,
                                         mode_,
                                         CUDNN_PROPAGATE_NAN,
                                         coef_);
        }

        void init(std::shared_ptr <Tensor> inputs) override {
            if (isInit) return;
            Layer::init(inputs);
            outputs_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), inputs_->h(), inputs_->w());
            inputGrads_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), inputs_->h(), inputs_->w());
            outputs_->valueInit();
            inputGrads_->valueInit();
        }

        std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> inputs) override {
            init(inputs);

            cudnnActivationForward(cudaCtx_->cudnn_,
                                   actDesc_,
                                   &cudaCtx_->one, inputs_->desc_, inputs_->gpu(),
                                   &cudaCtx_->one, outputs_->desc_, outputs_->gpu());

            return outputs_;
        }

        std::shared_ptr <Tensor> backward(std::shared_ptr <Tensor> grads) override {
            outputsGrads_ = std::move(grads);
            cudnnActivationBackward(cudaCtx_->cudnn_,
                                    actDesc_,
                                    &cudaCtx_->one,
                                    outputs_->desc_, outputs_->gpu(),
                                    outputsGrads_->desc_, outputsGrads_->gpu(),
                                    inputs_->desc_, inputs_->gpu(),
                                    &cudaCtx_->one,
                                    inputGrads_->desc_, inputGrads_->gpu());
            return inputGrads_;
        }

        ~Activation() {
            cudnnDestroyActivationDescriptor(actDesc_);
        }
    };

    class ReLU : public Activation {
    public:
        ReLU(float coef = 1.f) : Activation(coef, CUDNN_ACTIVATION_RELU) {
            name_ = "ReLU_" + std::to_string(globalID++);
        }
    };

    class BatchNorm: public Layer {
    public:
        std::shared_ptr<Tensor> gamma_ = nullptr;
        std::shared_ptr<Tensor> beta_ = nullptr;
        int channels_ = 0;

        BatchNorm(int channels): channels_(channels) {
            gamma_ = std::make_shared<Tensor>(1, 1, 1, channels);
            beta_ = std::make_shared<Tensor>(1, 1, 1, channels);
            gamma_->valueInit(1.f);
            beta_->valueInit(0.f);
        }

        void init(std::shared_ptr<Tensor> inputs) override {
            if (isInit) return;
            Layer::init(inputs);
            outputs_ = std::make_shared<Tensor>(inputs_);
        }

        std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> inputs) override {
            init(inputs);
            int size = outputs_->h() * outputs_->w();
            float alpha = 1 / (float) size, cmean, cvar, sttd;
            for (int c = 0; c < outputs_->c(); ++c) {
                cmean = 0.f;
                cvar = 0.f;
                for (int n = 0; n < outputs_->n(); ++n)
                    cmean += reduceSumVector(outputs_->gpu() + n * outputs_->len() + c * size,
                                    size,
                                    alpha);
                cmean /= (float) outputs_->n();
                for (int n = 0; n < outputs_->n(); ++n)
                    cvar += reduceSumVector(outputs_->gpu() + n * outputs_->len() + c * size,
                                            size,
                                            alpha,
                                            cmean,
                                            2.f);
                cvar /= (float) outputs_->n();
                sttd = pow(cvar, -0.5);
                for (int n = 0; n < outputs_->n(); ++n) {
                    vectorValueSum(sttd,
                                   outputs_->gpu() + n * outputs_->len() + c * size,
                                   sttd * cmean,
                                   size);
                }
            }

            return outputs_;
        }

        std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> grads) override {
            return nullptr;
        }

        ~BatchNorm() = default;
    };
};

#endif //MYNN_LAYER_H
