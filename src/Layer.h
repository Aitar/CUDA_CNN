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
# include "autograd.h"

namespace cuDL {
static int globalID = 0;

class Layer {
protected:
    tensorSp outputs_ = nullptr;
    tensorSp inputs_ = nullptr;
    tensorSp inputGrads_ = nullptr;
    tensorSp outputsGrads_ = nullptr;
    nodeSp outputNode_ = nullptr;
    nodeSp inputNode_ = nullptr;

    std::shared_ptr <CudaContext> cuda_ = nullptr;
    std::map <std::shared_ptr<Tensor>, std::shared_ptr<Tensor>> parmas_;

    std::string name_;
    int batchSize_ = 0;
    bool isInit = false;
    bool hasParams_ = false;

    friend class Module;

public:
    virtual tensorSp forward(tensorSp inputs) = 0;

    virtual tensorSp backward(tensorSp grads) = 0;

    virtual void makeGraph(nodeSp preNode) = 0;

    virtual void init(tensorSp inputs) {
        isInit = true;
        inputs_ = std::move(inputs);
        batchSize_ = inputs_->n();
        inputGrads_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), inputs_->h(), inputs_->w());
        inputGrads_->valueInit();
    };

    virtual void updateParams(float lr) {
        if (!hasParams_) {
            WARN("[Warning] Try to update parameters at layer without parameters.\n");
            return;
        }
        for (auto &param: parmas_)
            vectorAdd(lr, param.second->gpu(), 1.f, param.first->gpu(), param.first->gpu(), param.first->size());
        cudaDeviceSynchronize();
    }

    virtual void zeroGrad() {
        if (!hasParams_) {
            printf("[Warning] Try to zeroGrad at layer without parameters.\n");
            return;
        }
        for (auto &param: parmas_)
            param.second->valueInit();

        cudaDeviceSynchronize();
    }

    std::string getName() { return name_; };

    void setCudaContext(std::shared_ptr <CudaContext> cudaCtx) {
        cuda_ = std::move(cudaCtx);
    }

    virtual void print() {}

    nodeSp getOutputNode() {
        return NIL(outputNode_);
    }
};

class Linear : public Layer {
public:
    tensorSp weight_ = nullptr;  // (1, 1, inSize_, outSize_)
    tensorSp bias_ = nullptr;
    tensorSp wGrad_ = nullptr;
    tensorSp bGrad_ = nullptr;
    idenSp weightNode_ = nullptr;
    idenSp biasNode_ = nullptr;
    nodeSp gemmNode_ = nullptr;

    int inSize_ = 0;
    int outSize_ = 0;

    Linear(int inSize, int outSize) {
        name_ = "Linear_" + std::to_string(globalID++);
        inSize_ = inSize;
        outSize_ = outSize;
        hasParams_ = true;
    }

    ~Linear() = default;

    void init(tensorSp inputs) override {
        if (isInit) return;
        Layer::init(inputs);

        weight_ = std::make_shared<Tensor>(inSize_, outSize_);
        wGrad_ = std::make_shared<Tensor>(inSize_, outSize_);
        bias_ = std::make_shared<Tensor>(outSize_);
        bGrad_ = std::make_shared<Tensor>(outSize_);

        weight_->uniformInit();
        bias_->valueInit();
        wGrad_->valueInit();
        bGrad_->valueInit();
        outputs_ = std::make_shared<Tensor>(bias_, true, batchSize_);

        parmas_[weight_] = wGrad_;
        parmas_[bias_] = bGrad_;
    }

    tensorSp forward(tensorSp inputs) override {
        if (inputs->len() != inSize_) {
            stringstream ss;
            ss << "\nShape dismatch, expect " << inSize_ << "but got" << inputs->size() << endl;
            ERR(ss.str());
            exit(-1);
        }

        init(inputs);

            inputs_->reshape(1, 1, inputs_->c() * inputs_->h() * inputs_->w());

        // YT = WT * XT + BT
        // WT: O * I, XT: I * N, YT = O * N
        CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                   CUBLAS_OP_N, CUBLAS_OP_N,
                                   outSize_, batchSize_, inSize_,
                                   &cuda_->one,
                                   weight_->gpu(), outSize_,
                                   inputs_->gpu(), inSize_,
                                   &cuda_->one,
                                   outputs_->gpu(), outSize_));

//            printf("success!\n");
        return outputs_;
    }

    void makeGraph(nodeSp preNode) override {
        inputNode_ = std::move(NIL(preNode));
        init(inputNode_->getValue());
        weightNode_ = make_shared<Identity>(weight_, wGrad_);
        biasNode_ = make_shared<Identity>(bias_, bGrad_);
        outputNode_ = make_shared<Gemm>(nodes{inputNode_, weightNode_});
//        auto expand = make_shared<Expand>(nodes{biasNode_}, inputNode_->getValue()->n());
//        outputNode_ = make_shared<Add>(nodes{gemmNode_, expand}, 1, 1);
    }

    tensorSp backward(tensorSp grads) override {
        // db = dy
        outputsGrads_ = std::move(grads);
        bGrad_ = outputsGrads_;

        // dw = x * (dy)^T
        CUBLASCHECK(cublasSger_v2(cuda_->cublas_,
                                  outSize_, inSize_,
                                  &cuda_->one,
                                  outputsGrads_->gpu(), 1,      // dy: 1, 1, 1, outSize_
                                  inputs_->gpu(), 1,            // x:  1, 1, 1, inSize_
                                  wGrad_->gpu(), outSize_));     // dw: 1, 1, outSize_, inSize_

        // dx = w * (dy)^T
        // y = alpha * A @ x + beta * y
        CUBLASCHECK(cublasSgemv(cuda_->cublas_,
                                CUBLAS_OP_T,
                                outSize_, inSize_,
                                &cuda_->one,
                                weight_->gpu(), outSize_,
                                outputsGrads_->gpu(), 1,    // dy: (1, 1, 1, outSize_)
                                &cuda_->zero,
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
    tensorSp kernel_ = nullptr;
    tensorSp bias_ = nullptr;
    tensorSp kernelGrads_ = nullptr;
    tensorSp biasGrads_ = nullptr;
    nodeSp kernelNode_ = nullptr;
    nodeSp biasNode_ = nullptr;
    nodeSp convNode_ = nullptr;
    nodeSp addNode_ = nullptr;

    cudnnFilterDescriptor_t filterDesc_{};
    cudnnConvolutionDescriptor_t convDesc_{};

    cudnnConvolutionFwdAlgo_t fwdAlgo_;
    cudnnConvolutionBwdDataAlgo_t bwdDataAlgo_;
    cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo_;

    void** workspace_ = nullptr;
    size_t workspaceSize_ = 0;

public:
    std::string name_;
    int inChannels_;
    int outChannels_;
    int kernelSize_;
    int stride_;
    int padding_;
    int dilation_;
    int inputH_;
    int inputW_;
    int outputH_;
    int outputW_;


    Conv2D(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0,
           float paddingFill = 0.f, int dilation = 1) :
            inChannels_(inChannels),
            outChannels_(outChannels),
            kernelSize_(kernelSize),
            stride_(stride),
            padding_(padding),
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
        CUDNNCHECK(cudnnGetConvolutionForwardAlgorithmMaxCount(NIL(cuda_)->cudnn_, &maxAlgoCnt));
        CUDNNCHECK(cudnnGetConvolutionForwardAlgorithm_v7(NIL(cuda_)->cudnn_,
                                                          inputs_->desc_,
                                                          filterDesc_,
                                                          convDesc_,
                                                          outputs_->desc_,
                                                          maxAlgoCnt,
                                                          &algoCnt,
                                                          &fwdPerfs[0]));
        fwdAlgo_ = fwdPerfs[0].algo;
        CUDNNCHECK(cudnnGetConvolutionForwardWorkspaceSize(cuda_->cudnn_,
                                                           inputs_->desc_,
                                                           filterDesc_,
                                                           convDesc_,
                                                           outputs_->desc_,
                                                           fwdAlgo_, &tempSize));
        workspaceSize_ = std::max(workspaceSize_, tempSize);

        // bias 找到最合适的反向传播计算算法，
        CUDNNCHECK(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cuda_->cudnn_, &maxAlgoCnt));
        CUDNNCHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cuda_->cudnn_,
                                                                 inputs_->desc_,
                                                                 outputs_->desc_,
                                                                 convDesc_,
                                                                 filterDesc_,
                                                                 maxAlgoCnt,
                                                                 &algoCnt,
                                                                 &bwdFilterPerfs[0]));
        bwdFilterAlgo_ = bwdFilterPerfs[0].algo;
        CUDNNCHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(cuda_->cudnn_,
                                                                  inputs_->desc_,
                                                                  outputs_->desc_,
                                                                  convDesc_,
                                                                  filterDesc_,
                                                                  bwdFilterAlgo_, &tempSize));
        workspaceSize_ = std::max(workspaceSize_, tempSize);

        // data - bwd
        CUDNNCHECK(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cuda_->cudnn_, &maxAlgoCnt));
        CUDNNCHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(cuda_->cudnn_,
                                                               filterDesc_,
                                                               outputs_->desc_,
                                                               convDesc_,
                                                               inputs_->desc_,
                                                               maxAlgoCnt,
                                                               &algoCnt,
                                                               &bwdDataPerfs[0]));
        bwdDataAlgo_ = bwdDataPerfs[0].algo;
        CUDNNCHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(cuda_->cudnn_,
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
    }

    void init(tensorSp inputs) override {
        if (isInit) return;

        Layer::init(NIL(inputs));
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

        setWorkspace();
    }

    tensorSp forward(tensorSp inputs) override {
        init(inputs);
        CUDNNCHECK(cudnnConvolutionForward(cuda_->cudnn_,
                                           &cuda_->one, inputs_->desc_, inputs_->gpu(),
                                           filterDesc_, kernel_->gpu(),
                                           convDesc_, fwdAlgo_,
                                           workspace_, workspaceSize_,
                                           &cuda_->zero, outputs_->desc_, outputs_->gpu()));

        CUDNNCHECK(cudnnAddTensor(cuda_->cudnn_,
                                  &cuda_->one, bias_->desc_, bias_->gpu(),
                                  &cuda_->one, outputs_->desc_, outputs_->gpu()));
        return outputs_;
    }

    void makeGraph(nodeSp preNode) override {
        inputNode_ = std::move(preNode);
        init(inputNode_->getValue());
        kernelNode_ = make_shared<Identity>(kernel_, kernelGrads_);
        biasNode_ = make_shared<Identity>(bias_, biasGrads_);
        outputNode_ = make_shared<Conv>(
                nodes{inputNode_, kernelNode_},
                outputH_,
                outputW_,
                filterDesc_,
                convDesc_,
                fwdAlgo_,
                bwdDataAlgo_,
                bwdFilterAlgo_,
                workspace_,
                workspaceSize_);
    }


    tensorSp backward(tensorSp grads) override {
        outputsGrads_ = std::move(grads);

        CUDNNCHECK(cudnnConvolutionBackwardBias(cuda_->cudnn_,
                                                &cuda_->one, outputsGrads_->desc_, outputsGrads_->gpu(),
                                                &cuda_->zero, biasGrads_->desc_, biasGrads_->gpu()));

        CUDNNCHECK(cudnnConvolutionBackwardFilter(cuda_->cudnn_,
                                                  &cuda_->one, inputs_->desc_, inputs_->gpu(),
                                                  outputsGrads_->desc_, outputsGrads_->gpu(),
                                                  convDesc_, bwdFilterAlgo_,
                                                  workspace_, workspaceSize_,
                                                  &cuda_->zero, filterDesc_, kernelGrads_->gpu()));

        CUDNNCHECK(cudnnConvolutionBackwardData(cuda_->cudnn_,
                                                &cuda_->one, filterDesc_, kernel_->gpu(),
                                                outputsGrads_->desc_, outputsGrads_->gpu(),
                                                convDesc_, bwdDataAlgo_,
                                                workspace_, workspaceSize_,
                                                &cuda_->zero, inputGrads_->desc_, inputGrads_->gpu()));
        return inputGrads_;
    }
};

class Pooling2D : public Layer {
private:
    int kernelSize_;
    int padding_;
    int stride_;
    int inputH_;
    int inputW_;
    int outputH_;
    int outputW_;
    cudnnPoolingMode_t mode_;
    cudnnPoolingDescriptor_t poolingDesc_;
    nodeSp poolingNode_ = nullptr;

public:
    Pooling2D(int kernelSize, int padding, int stride, cudnnPoolingMode_t mode) :
            kernelSize_(kernelSize), padding_(padding), stride_(stride), mode_(mode) {

        cudnnCreatePoolingDescriptor(&poolingDesc_);
        cudnnSetPooling2dDescriptor(poolingDesc_,
                                    mode_, CUDNN_PROPAGATE_NAN,
                                    kernelSize_, kernelSize_,
                                    padding_, padding_,
                                    stride_, stride_);
    }

    ~Pooling2D() {
        cudnnDestroyPoolingDescriptor(poolingDesc_);
    }

    void init(tensorSp inputs) override {
        if (isInit) return;

        Layer::init(inputs);
        inputH_ = inputs->h();
        inputW_ = inputs->w();
        outputH_ = (inputH_ + 2 * padding_ - kernelSize_) / stride_ + 1;
        outputW_ = (inputW_ + 2 * padding_ - kernelSize_) / stride_ + 1;
        outputs_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), outputH_, outputW_);
        outputsGrads_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), outputH_, outputW_);
        outputs_->valueInit();
        outputsGrads_->valueInit();
    }

    tensorSp forward(tensorSp inputs) override {
        init(inputs);
        CUDNNCHECK(cudnnPoolingForward(cuda_->cudnn_, poolingDesc_,
                                       &cuda_->one, inputs_->desc_, inputs_->gpu(),
                                       &cuda_->zero, outputs_->desc_, outputs_->gpu()));
        return outputs_;
    }

    tensorSp backward(tensorSp grads) override {
        outputsGrads_ = std::move(grads);
        CUDNNCHECK(cudnnPoolingBackward(cuda_->cudnn_, poolingDesc_,
                                        &cuda_->one,
                                        outputs_->desc_, outputs_->gpu(),
                                        outputsGrads_->desc_, outputsGrads_->gpu(),
                                        inputs_->desc_, inputs_->gpu(),
                                        &cuda_->zero,
                                        inputGrads_->desc_, inputGrads_->gpu()));

        return inputGrads_;
    }

    void makeGraph(nodeSp preNode) override {
        inputNode_ = std::move(preNode);
        init(inputNode_->getValue());
        outputNode_ = make_shared<Pooling>(nodes{inputNode_},
                                           inputs_,
                                           inputGrads_,
                                           outputs_,
                                           outputsGrads_,
                                           poolingDesc_);
    }
};

class MaxPooling : public Pooling2D {
public:
    explicit MaxPooling(int kernelSize, int padding = 0, int stride = 1) :
            Pooling2D(kernelSize, padding, stride, CUDNN_POOLING_MAX) {
        name_ = "MaxPooling_" + std::to_string(globalID++);
    }
};

class AvgPooling : public Pooling2D {
public:
    explicit AvgPooling(int kernelSize = 2, int padding = 0, int stride = 1) :
            Pooling2D(kernelSize, padding, stride, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING) {
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
    tensorSp loss_;
    nodeSp exp_;
    nodeSp sum_;
    nodeSp reciprocal_;
    nodeSp expand_;


    Softmax() {
        name_ = "Softmax_" + std::to_string(globalID++);
    }

    void init(tensorSp inputs) override {
        if (isInit) return;

        Layer::init(inputs);
        size_ = inputs->w();
        outputs_ = std::make_shared<Tensor>(batchSize_, 1, 1, size_);
        inputGrads_ = std::make_shared<Tensor>(batchSize_, 1, 1, size_);
        loss_ = std::make_shared<Tensor>(1);
        outputs_->valueInit();
        loss_->valueInit();
    }

    tensorSp forward(tensorSp inputs) override {
        init(inputs);
        CUDNNCHECK(cudnnSoftmaxForward(cuda_->cudnn_,
                                       CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                       &cuda_->one, inputs_->desc_, inputs_->gpu(),
                                       &cuda_->zero, outputs_->desc_, outputs_->gpu()));

        return outputs_;
    }

    void makeGraph(nodeSp preNode) override {
        inputNode_ = std::move(preNode);
        init(inputNode_->getValue());
        outputNode_ = make_shared<SoftmaxOp>(nodes{inputNode_});
    }

    float getLoss(const tensorSp &labels) {
        float loss = 0.f;
        for (int i = 0; i < outputs_->n(); ++i) {
            dSoftmax(inputGrads_->gpu() + i * outputs_->len(),
                     outputs_->gpu() + i * outputs_->len(),
                     batchSize_,
                     size_,
                     (int) labels->cpu()[i]);
            loss -= std::log(outputs_->getItem(i, 0, 0, (int) labels->cpu()[i]) + 1e-5);
        }
        return loss / (float) outputs_->n();
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

    tensorSp backward(tensorSp grads) override {
        if (grads != nullptr) {
            printf("[Warning] Softmax backward only accept nullptr.\n");
            return nullptr;
        }
        return inputGrads_;
    }

    void print() override {
        printf("\n%s", name_.c_str());
        printf("\noutputsGrads:");
        if (outputsGrads_ != nullptr) outputsGrads_->print();
        printf("\ninputGrads:");
        if (inputGrads_ != nullptr) inputGrads_->print();
        else printf("not init yet\n");
    }
};

class CrossEntropy: Layer {
private:
    nodeSp labelNode_ = nullptr;
    nodeSp logitsNode_ = nullptr;
    shared_ptr<Executor> executor_ = nullptr;

    void init(tensorSp inputs) override {}

    tensorSp forward(tensorSp inputs) override {
        return nullptr;
    }

    tensorSp backward(cuDL::tensorSp grads) override {
        return nullptr;
    }

    void makeGraph(nodeSp preNode) override {}

public:
    CrossEntropy(shared_ptr<CudaContext> cuda) {
        name_ = "CrossEntropy_" + std::to_string(globalID++);
        cuda_ = std::move(cuda);
        labelNode_ = make_shared<Identity>();
        logitsNode_ = make_shared<Identity>();
    }

    void init() {
        auto ln = make_shared<Ln>(nodes{logitsNode_});
        auto mul = make_shared<Mul>(nodes{ln, labelNode_});
        outputNode_ = make_shared<Sum>(nodes{mul}, -1);
        executor_ = make_shared<Executor>(outputNode_, cuda_);
    }

    tensorSp getLoss(tensorSp logits, tensorSp labels) {
        labelNode_->setValue(std::move(labels));
        logitsNode_->setValue(std::move(logits));
        if (!isInit) init();
        return executor_->forward();
    }

    tensorSp backward() {
        if (inputGrads_ == nullptr) {
            inputGrads_ = make_shared<Tensor>(1);
            inputGrads_->valueInit(1);
        }
        executor_->backward(inputGrads_);
        return logitsNode_->getGrad();
    }
};


class Activation : public Layer {
private:
    cudnnActivationDescriptor_t actDesc_;
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

    void init(tensorSp inputs) override {
        if (isInit) return;
        Layer::init(inputs);
        outputs_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), inputs_->h(), inputs_->w());
        inputGrads_ = std::make_shared<Tensor>(inputs_->n(), inputs_->c(), inputs_->h(), inputs_->w());
        outputs_->valueInit();
        inputGrads_->valueInit();
    }

    tensorSp forward(tensorSp inputs) override {
        init(inputs);

        cudnnActivationForward(cuda_->cudnn_,
                               actDesc_,
                               &cuda_->one, inputs_->desc_, inputs_->gpu(),
                               &cuda_->one, outputs_->desc_, outputs_->gpu());

        return outputs_;
    }

    tensorSp backward(tensorSp grads) override {
        outputsGrads_ = std::move(grads);
        cudnnActivationBackward(cuda_->cudnn_,
                                actDesc_,
                                &cuda_->one,
                                outputs_->desc_, outputs_->gpu(),
                                outputsGrads_->desc_, outputsGrads_->gpu(),
                                inputs_->desc_, inputs_->gpu(),
                                &cuda_->one,
                                inputGrads_->desc_, inputGrads_->gpu());
        return inputGrads_;
    }

    void makeGraph(nodeSp preNode) override {
        inputNode_ = std::move(preNode);
        init(inputNode_->getValue());
        outputNode_ = make_shared<Active>(nodes{inputNode_},
                                          inputs_,
                                          inputGrads_,
                                          outputs_,
                                          outputsGrads_,
                                          actDesc_);
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
private:

    nodeSp alphaNode_ = nullptr;
    nodeSp betaNode_ = nullptr;

public:
    std::shared_ptr<Tensor> alpha_ = nullptr;
    std::shared_ptr<Tensor> beta_ = nullptr;

    BatchNorm() {
        alpha_ = std::make_shared<Tensor>(1);
        beta_ = std::make_shared<Tensor>(1);
        alpha_->valueInit(1.f);
        beta_->valueInit(0.f);
    }

    void init(std::shared_ptr<Tensor> inputs) override {
        if (isInit) return;
        Layer::init(inputs);
        outputs_ = std::make_shared<Tensor>(inputs_, true);
        outputsGrads_ = std::make_shared<Tensor>(outputs_);
    }

    std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> inputs) override {
        return outputs_;
    }

    std::shared_ptr<Tensor> backward(std::shared_ptr<Tensor> grads) override {
        return nullptr;
    }

    void makeGraph(nodeSp preNode) override {
        inputNode_ = std::move(preNode);
        init(inputNode_->getValue());
        int n = inputNode_->getValue()->size();
        /**                                      a            β -> expand
         *                                       |                    |
         *  / sum -> expand \    / sum -> pow -> mul -> expand \      |
         * x ---------------> add ---------------------------> mul-> add -> out
         */
        alphaNode_ = make_shared<Identity>(alpha_);
        betaNode_ = make_shared<Identity>(beta_);
        auto mean = make_shared<Sum>(nodes{inputNode_}, 1.f / n);
        auto meanVector = make_shared<Expand>(nodes{mean}, n);
        auto xSubMean = make_shared<Add>(nodes{inputNode_, meanVector}, 1.f, -1);
        auto var = make_shared<Sum>(nodes{xSubMean}, 1.f / n, 0.f, 2.f);
        auto reStd = make_shared<Pow>(nodes{var}, -0.5);
        auto alpha = make_shared<Mul>(nodes{alphaNode_, reStd});
        auto aVec = make_shared<Expand>(nodes{alpha}, n);
        auto norm = make_shared<Mul>(nodes{xSubMean, aVec});
        auto bVec = make_shared<Expand>(nodes{betaNode_}, n);
        outputNode_ = make_shared<Add>(nodes{norm, bVec});
    }

    ~BatchNorm() = default;
};
};

#endif //MYNN_LAYER_H
