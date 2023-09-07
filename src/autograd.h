#ifndef CUDA_CNN_AUTOGRAD_H
#define CUDA_CNN_AUTOGRAD_H

# include <string>
# include <utility>
# include <set>
# include "utils.cuh"
# include "kernels.cuh"
#include <iostream>
#include <iostream>
#include <chrono>

namespace cuDL{
    using namespace std;
    static int globalNodeID = 0;

    const float GRAD_MIN = 1.5;
    const float GRAD_MAX = 10;

    void gradNrom(tensorSp grad) {
        pullTo(grad->gpu(), GRAD_MIN, GRAD_MAX, grad->size());
    }

    class OpNode{
    protected:
        tensorSp value_ = nullptr;
        tensorSp grad_ = nullptr;
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> b_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        shared_ptr<Tensor> db_ = nullptr;
        shared_ptr<CudaContext> cuda_ = nullptr;

        tensors inputGrads_;
        tensors params_;

        bool isInit_ = false;
        bool noGrad_ = false;

    public:
        vector<shared_ptr<OpNode>> inNodes_;
        string name_;
        
        virtual void forward() = 0;
        virtual tensors& backward() = 0;

        void setName(const string& name) {
            name_ = name + "_" + std::to_string(globalNodeID++);
        }

        tensorSp getValue() {
            return value_;
        }

        void setValue(tensorSp value, bool noGrad=false) {
            value_ = std::move(value);
            noGrad_ = noGrad;
            if (!noGrad) {
                if (grad_ == nullptr || grad_->size() != value_->size()) {
                    grad_ = make_shared<Tensor>(value_);
                    WARN("Value shape has changed, grad re-malloced.");
                } else grad_->valueInit();
            }
        }

        tensorSp getGrad() {
            return grad_;
        }

        void setGrad(tensorSp grad) {
            if (noGrad_) {
                WARN("Set grad for a node with noGrad = true.");
            }
            grad_ = std::move(grad);
        }

        void zeroGrad() {
            grad_->valueInit();
        }


        void setCuda(shared_ptr<CudaContext> cuda) {
            cuda_ = std::move(cuda);
        }

        vector<shared_ptr<OpNode>>& getInputNodes() {
            return inNodes_;
        }

        virtual void updateValue(float lr) {}

        virtual void print() {
            cout << name_ << endl;
            cout << "input:" << endl;
            a_->print();
            cout << "------------------------" << endl;
            if (b_ != nullptr) b_->print();
            cout << "output:" << endl;
            value_->print();
            cout << "grad:" << endl;
            cout << "dy:" << endl;
            grad_->print();
            cout << "da:" << endl;
            da_->print();
            if (db_ != nullptr) {
                cout << "db:" << endl;
                db_->print();
            }

        };
    };

    typedef vector<shared_ptr<OpNode>> nodes;


    class Add: public OpNode {
    private:
        float alpha_ = 1.f;
        float beta_ = 1.f;

    public:
        Add(vector<shared_ptr<OpNode>> inNodes) {

            setName("Add");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        Add(vector<shared_ptr<OpNode>> inNodes,
            float alpha,
            float beta): alpha_(alpha), beta_(beta) {

            setName("Add");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());

            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            b_ = NIL(NIL(inNodes_[1])->getValue());
            da_ = make_shared<Tensor>(a_);
            db_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
            inputGrads_.push_back(db_);
            value_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(value_);
        }

        void forward() override {
            if (!isInit_) init();
            cudaDeviceSynchronize();

            vectorAdd(alpha_, a_->gpu(), beta_, b_->gpu(), value_->gpu(), value_->size());
        }

        tensors& backward() override {
            da_->copy(grad_);
            db_->copy(grad_);
            CUBLASCHECK(cublasSscal_v2(cuda_->cublas_, da_->size(), &alpha_,  da_->gpu(), 1));
            CUBLASCHECK(cublasSscal_v2(cuda_->cublas_, db_->size(), &beta_,  db_->gpu(), 1));

            gradNrom(da_);
            gradNrom(db_);
            return inputGrads_;
        }
    };

    class Gemm: public OpNode {
    public:
        Gemm(nodes inNodes) {
            setName("Gemm");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;

            a_ = NIL(NIL(inNodes_[0])->getValue());
            b_ = NIL(NIL(inNodes_[1])->getValue());
            da_ = make_shared<Tensor>(a_->n(), a_->len());
            db_ = make_shared<Tensor>(b_->n(),  b_->len());
            inputGrads_.push_back(da_);
            inputGrads_.push_back(db_);
            value_ = make_shared<Tensor>(a_->n(), b_->len());
            grad_ = make_shared<Tensor>(a_->n(), b_->len());
        }

        void forward() override {
            if (!isInit_) init();
            if (a_->len() != b_->n()) {
                stringstream ss;
                ss << "Gemm shape mismatch, A(" << a_->n() << ", " << a_->len() << "), B(" << b_->n() << ", " << b_->len() << ").";
                ERR(ss.str());
                exit(-1);
            }

            CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       b_->len(), a_->n(), a_->len(),
                                       &cuda_->one,
                                       b_->gpu(), b_->len(),
                                       a_->gpu(), a_->len(),
                                       &cuda_->zero,
                                       value_->gpu(), b_->len()));
            cudaDeviceSynchronize();
        }

        tensors& backward() override {

            // dA = dY * B^T
            auto A = b_;
            auto B = grad_;
            auto C = da_;
            // C = B * A^T
            CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       A->n(), B->n(), A->len(),
                                       &cuda_->one,
                                       A->gpu(), A->len(),
                                       B->gpu(), B->len(),
                                       &cuda_->zero,
                                       C->gpu(), C->len()));

            // dB = A^T * dY
            A = a_;
            C = db_;
            // C = A^T * B, C^T = B^T * A
            CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                       CUBLAS_OP_N, CUBLAS_OP_T,
                                       B->len(), A->len(), B->n(),
                                       &cuda_->one,
                                       B->gpu(), B->len(),
                                       A->gpu(), A->len(),
                                       &cuda_->zero,
                                       C->gpu(), C->len()));
            gradNrom(da_);
            gradNrom(db_);
            return inputGrads_;
        }

    };

    class Mul: public OpNode {
    public:
        Mul(nodes inNodes) {
            setName("Mul");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            b_ = NIL(NIL(inNodes_[1])->getValue());
            da_ = make_shared<Tensor>(a_);
            db_ = make_shared<Tensor>(b_);
            inputGrads_.push_back(da_);
            inputGrads_.push_back(db_);
            value_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(a_);
        }

        void forward() override {
            if (!isInit_) init();
            cudaDeviceSynchronize();
            hadamardPorduct(cuda_->one, a_->size(), a_->gpu(), b_->gpu(), value_->gpu());
        }

        tensors& backward() override {
            hadamardPorduct(cuda_->one, grad_->size(), grad_->gpu(), b_->gpu(), da_->gpu());
            hadamardPorduct(cuda_->one, grad_->size(), grad_->gpu(), a_->gpu(), db_->gpu());
            gradNrom(da_);
            gradNrom(db_);
            return inputGrads_;
        }
    };

    class Sum: public OpNode {
    private:
        float alpha_ = 1.f;
        float bias_ = 0.f;
        float p_ = 1.f;

    public:
        Sum(nodes inNodes, float alpha = 1.f, float bias = 0.f, float p = 1.f)
        : alpha_(alpha), bias_(bias), p_(p) {
            setName("Sum");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            value_ = make_shared<Tensor>(1);
            grad_ = make_shared<Tensor>(1);
            da_ = make_shared<Tensor>(a_, true);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            cudaDeviceSynchronize();
            vectorSum(a_->gpu(), value_->gpu(), a_->size(), alpha_, bias_, p_);
        }

        tensors& backward() override {
            float a = grad_->cpu()[0] * alpha_ * p_;
            vectorValueAdd(alpha_, da_->gpu(), da_->size(), bias_);
            pow(da_->gpu(), da_->gpu(), p_ - 1, da_->size());
            CUBLASCHECK(cublasSscal_v2(cuda_->cublas_, da_->size(),  &a, da_->gpu(), 1));
            gradNrom(da_);
            return inputGrads_;
        }

        void print() {
            cout << name_ << endl;
            cout << "a: " << alpha_ << ", β: " << bias_ << endl;
            cout << "input:" << endl;
            a_->print();
            cout << "output:" << endl;
            value_->print();
            cout << "grad:" << endl;
            grad_->print();
            cout << "da:" << endl;
            da_->print();
        }
    };

    class Exp: public OpNode {
    public:
        Exp(nodes inNodes) {
            setName("Exp");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            da_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(a_);
            value_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            if (a_->size() < 8 && a_->device() == CPU) {
                for (int i = 0; i < a_->size(); ++i)
                    value_->cpu()[i] = std::exp(a_->cpu()[i]);
            } else {
                exp(a_->gpu(), value_->gpu(), a_->size());
            }
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            hadamardPorduct(1, grad_->size(), value_->gpu(), grad_->gpu(), da_->gpu());
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Ln: public OpNode {
    public:
        Ln(nodes inNodes) {
            setName("Ln");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            da_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(a_);
            value_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            if (a_->size() < 8 && a_->device() == CPU) {
                for (int i = 0; i < a_->size(); ++i)
                    value_->cpu()[i] = std::log(a_->cpu()[i]);
            } else {
                log(a_->gpu(), value_->gpu(), a_->size());
            }
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            pow(a_->gpu(), da_->gpu(), -1, a_->size());
            hadamardPorduct(1, da_->size(), da_->gpu(), grad_->gpu(), da_->gpu());
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Expand: public OpNode {
    private:
        int dup_;

    public:
        Expand(nodes inNodes, int dup, tensorSp tensor=nullptr): dup_(dup) {
            setName("Expand");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init(tensorSp tensor=nullptr) {
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            da_ = make_shared<Tensor>(a_);
            if (tensor != nullptr) value_ = make_shared<Tensor>(tensor);
            else value_ = make_shared<Tensor>(a_->n(), a_->c(), a_->h(), a_->w() * dup_);
            grad_ = make_shared<Tensor>(value_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            cudaDeviceSynchronize();
            value_->copy(a_, dup_);
        }

        tensors& backward() override {
            vectorSum(grad_->gpu(), da_->gpu(), grad_->size(), 1.f / (float) grad_->size());
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Split: public OpNode {
    private:
        int part_;

    public:
        Split(nodes inNodes, int part): part_(part) {
            setName("Split");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            da_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(a_);
            value_ = make_shared<Tensor>(grad_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            da_->cpu()[0] = grad_->cpu()[0];
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Combine: public OpNode {
    public:
        Combine(nodes inNodes) {
            setName("Combine");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;

            grad_ = make_shared<Tensor>(a_->n(), a_->c(), a_->h(), a_->w());
            value_ = make_shared<Tensor>(grad_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            value_->copy(a_);
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            da_->cpu()[0] = grad_->cpu()[0];
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Pow: public OpNode {
    private:
        float p_;

    public:
        Pow(nodes inNodes, float p) {
            setName("Pow");
            p_ = p;
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            a_ = NIL(NIL(inNodes_[0])->getValue());
            da_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(a_);
            value_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            cudaDeviceSynchronize();

            if (a_->size() < 8 && a_->device() == CPU) {
                for (int i = 0; i < a_->size(); ++i)
                    value_->cpu()[i] = p_ < 0 ? std::pow(a_->cpu()[i] + 1e-9, p_) : std::pow(a_->cpu()[i], p_);
            } else {
                pow(a_->gpu(), value_->gpu(), p_, a_->size());
            }
        }

        tensors& backward() override {
            hadamardPorduct(1, value_->size(), value_->gpu(), value_->gpu(), da_->gpu());
            hadamardPorduct(-1, grad_->size(), value_->gpu(), grad_->gpu(), da_->gpu());
            gradNrom(da_);
            return inputGrads_;
        }

        void print() override {
            cout << name_ << endl;
            cout << "p: " << p_ << endl;
            cout << "input:" << endl;
            a_->print();
            cout << "output:" << endl;
            value_->print();
            cout << "grad:" << endl;
            cout << "dy:" << endl;
            grad_->print();
            cout << "da:" << endl;
            da_->print();
        }
    };

    class Conv: public OpNode {
    private:
        int outputH_;
        int outputW_;

        shared_ptr<Tensor> x_ = nullptr;
        shared_ptr<Tensor> w_ = nullptr;
        shared_ptr<Tensor> dx_ = nullptr;
        shared_ptr<Tensor> dw_ = nullptr;

        cudnnFilterDescriptor_t filterDesc_;
        cudnnConvolutionDescriptor_t convDesc_;

        cudnnConvolutionFwdAlgo_t fwdAlgo_;
        cudnnConvolutionBwdDataAlgo_t bwdDataAlgo_;
        cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo_;

        void** workspace_;
        size_t workspaceSize_ = 0;

    public:
        Conv(nodes inNodes,
             int outputH,
             int outputW,
             cudnnFilterDescriptor_t filterDesc,
             cudnnConvolutionDescriptor_t convDesc,
             cudnnConvolutionFwdAlgo_t fwdAlgo,
             cudnnConvolutionBwdDataAlgo_t bwdDataAlgo,
             cudnnConvolutionBwdFilterAlgo_t bwdFilterAlgo,
             void **workspace,
             size_t workspaceSize):

             outputH_(outputH),
             outputW_(outputW),
             filterDesc_(filterDesc),
             convDesc_(convDesc),
             fwdAlgo_(fwdAlgo),
             bwdDataAlgo_(bwdDataAlgo),
             bwdFilterAlgo_(bwdFilterAlgo),
             workspace_(workspace),
             workspaceSize_(workspaceSize) {

            setName("Conv");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;

            x_ = NIL(NIL(inNodes_[0])->getValue());
            w_ = NIL(NIL(inNodes_[1])->getValue());
            dx_ = make_shared<Tensor>(x_);
            dw_ = make_shared<Tensor>(w_);

            value_ = make_shared<Tensor>(x_->n(), w_->c(), outputH_, outputW_);
            grad_ = make_shared<Tensor>(x_->n(), w_->c(), outputH_, outputW_);

            inputGrads_.push_back(dx_);
            inputGrads_.push_back(dw_);
        }

        void forward() override {
            if (!isInit_) init();
            CUDNNCHECK(cudnnConvolutionForward(cuda_->cudnn_,
                                               &cuda_->one, x_->desc_, x_->gpu(),
                                               filterDesc_, w_->gpu(),
                                               convDesc_, fwdAlgo_,
                                               workspace_, workspaceSize_,
                                               &cuda_->zero, value_->desc_, value_->gpu()));
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            CUDNNCHECK(cudnnConvolutionBackwardFilter(cuda_->cudnn_,
                                                      &cuda_->one, x_->desc_, x_->gpu(),
                                                      grad_->desc_, grad_->gpu(),
                                                      convDesc_, bwdFilterAlgo_,
                                                      workspace_, workspaceSize_,
                                                      &cuda_->zero, filterDesc_, dw_->gpu()));

            CUDNNCHECK(cudnnConvolutionBackwardData(cuda_->cudnn_,
                                                    &cuda_->one, filterDesc_, w_->gpu(),
                                                    grad_->desc_, grad_->gpu(),
                                                    convDesc_, bwdDataAlgo_,
                                                    workspace_, workspaceSize_,
                                                    &cuda_->zero, dx_->desc_, dx_->gpu()));
            gradNrom(dx_);
            gradNrom(dw_);
            return inputGrads_;
        }

        void print() override {
            cout << name_ << endl;
            cout << "input:" << endl;
            w_->print();
            x_->print();
            cout << "output:" << endl;
            value_->print();
            cout << "grad:" << endl;
            cout << "dy:" << endl;
            grad_->print();
            cout << "dw:" << endl;
            dw_->print();
            cout << "dx:" << endl;
            dx_->print();
        }
    };

    class Pooling: public OpNode {
    private:
        int outputH_;
        int outputW_;
        cudnnPoolingDescriptor_t poolingDesc_;

    public:
        Pooling(nodes inNodes,
                int outputH,
                int outputW,
                cudnnPoolingDescriptor_t poolingDesc):

            outputH_(outputH),
            outputW_(outputW),
            poolingDesc_(poolingDesc) {
            setName("Pooling");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        Pooling(nodes inNodes,
                shared_ptr<Tensor> a,
                shared_ptr<Tensor> da,
                shared_ptr<Tensor> value,
                shared_ptr<Tensor> grad,
                cudnnPoolingDescriptor_t poolingDesc):
                poolingDesc_(poolingDesc) {

            setName("Pooling");
            a_ = std::move(a);
            da_ = std::move(da);
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            value_ = std::move(value);
            grad_ = std::move(grad);
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            if (a_ == nullptr) a_ = NIL(NIL(inNodes_[0])->getValue());
            if (da_ == nullptr) da_ = make_shared<Tensor>(a_);
            if (grad_ == nullptr) grad_ = make_shared<Tensor>(a_->n(), a_->c(), outputH_, outputW_);
            if (value_ == nullptr) value_ = make_shared<Tensor>(a_->n(), a_->c(), outputH_, outputW_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            CUDNNCHECK(cudnnPoolingForward(cuda_->cudnn_, poolingDesc_,
                                           &cuda_->one, a_->desc_, a_->gpu(),
                                           &cuda_->zero, value_->desc_, value_->gpu()));
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            CUDNNCHECK(cudnnPoolingBackward(cuda_->cudnn_, poolingDesc_,
                                            &cuda_->one,
                                            value_->desc_, value_->gpu(),
                                            grad_->desc_, grad_->gpu(),
                                            a_->desc_, a_->gpu(),
                                            &cuda_->zero,
                                            da_->desc_, da_->gpu()));
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Active: public OpNode {
    private:
        cudnnActivationDescriptor_t actDesc_;

    public:
        Active(nodes inNodes,
               shared_ptr<Tensor> a,
               shared_ptr <Tensor> da,
               shared_ptr <Tensor> value,
               shared_ptr<Tensor> grad,
               cudnnActivationDescriptor_t actDesc): actDesc_(actDesc) {
            a_ = std::move(a);
            da_ = std::move(da);
            grad_ = std::move(grad);
            value_ = std::move(value);
            setName("Active");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        Active(nodes inNodes, cudnnActivationDescriptor_t actDesc) {
            actDesc_ = actDesc;
            setName("Active");

            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        void init() {
            if (isInit_) return;
            isInit_ = true;
            if (a_ == nullptr) a_ = NIL(NIL(inNodes_[0])->getValue());
            if (da_ == nullptr) da_ = make_shared<Tensor>(a_);
            if (grad_ == nullptr) grad_ = make_shared<Tensor>(a_);
            if (value_ == nullptr) value_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            
            CUDNNCHECK(cudnnActivationForward(cuda_->cudnn_,
                                   actDesc_,
                                   &cuda_->one, a_->desc_, a_->gpu(),
                                   &cuda_->one, value_->desc_, value_->gpu()));
            cudaDeviceSynchronize();
        }

        tensors& backward() override {
            CUDNNCHECK(cudnnActivationBackward(cuda_->cudnn_,
                                    actDesc_,
                                    &cuda_->one,
                                    value_->desc_, value_->gpu(),
                                    grad_->desc_, grad_->gpu(),
                                    a_->desc_, a_->gpu(),
                                    &cuda_->one,
                                    da_->desc_, da_->gpu()));
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class SoftmaxOp: public OpNode {
    public:
        SoftmaxOp(nodes inNodes) {
            setName("Softmax");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }

        SoftmaxOp(nodes inNodes,
                  shared_ptr<Tensor> a,
                  shared_ptr <Tensor> da,
                  shared_ptr <Tensor> value,
                  shared_ptr<Tensor> grad) {

            grad_ = std::move(grad);
            value_ = std::move(value);
            a_ = std::move(a);
            da_ = std::move(da);
            setName("Softmax");

            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            init();
        }


        void init() {
            if (isInit_) return;
            isInit_ = true;
            if (a_ == nullptr) a_ = NIL(NIL(inNodes_[0])->getValue());
            if (da_ == nullptr) da_ = make_shared<Tensor>(a_);
            if (grad_ == nullptr) grad_ = make_shared<Tensor>(a_);
            if (value_ == nullptr) value_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit_) init();
            CUDNNCHECK(cudnnSoftmaxForward(cuda_->cudnn_,
                                           CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                           &cuda_->one, a_->desc_, a_->gpu(),
                                           &cuda_->zero, value_->desc_, value_->gpu()));
        }

        tensors& backward() override {
            CUDNNCHECK(cudnnSoftmaxBackward(cuda_->cudnn_,
                                            CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_INSTANCE,
                                            &cuda_->one, value_->desc_, value_->gpu(),
                                            grad_->desc_, grad_->gpu(),
                                            &cuda_->zero, da_->desc_, da_->gpu()));
            gradNrom(da_);
            return inputGrads_;
        }
    };

    class Identity: public OpNode {
    public:
        Identity() {
            setName("Identity");
        }

        Identity(tensorSp value) {
            setName("Identity");

            value_ = std::move(value);
            grad_ = make_shared<Tensor>(value_);
        }

        Identity(tensorSp value, tensorSp grad) {
            setName("Identity");

            value_ = std::move(value);
            grad_ = std::move(grad);
        }

        void forward() override { }

        tensors& backward() override {
            return inputGrads_;
        }

        void print() override {
            cout << name_ << endl;
            cout << "value:" << endl;
            value_->print();
            cout << "grad:" << endl;
            grad_->print();
        }

        void updateValue(float lr) override {
            if (noGrad_) WARN("Try to update a identity with noGrad == true.");
//            lr *= -1;
            gradNrom(grad_);
//            cout << "grad:" << endl;
//            grad_->print();
//            cout << "before update:" << endl;
//            value_->print();
            vectorAdd(1.f, value_->gpu(), lr, grad_->gpu(), value_->gpu(), value_->size());
//            cout << "after update:" << endl;
//            value_->print();
            if (value_->size() > 1) {
                norm(value_);
//                cout << "after norm:" << endl;
//                value_->print();
            }
            cudaDeviceSynchronize();
        }
    };

    class Executor {
    private:
        shared_ptr<OpNode> root_;
        shared_ptr<CudaContext> cuda_;
        nodes backwordTopo;
        nodes forwordTopo;
        float lr_;

    public:
        Executor(shared_ptr<OpNode> root, shared_ptr<CudaContext> cuda):
            root_(std::move(root)), cuda_(std::move(cuda)) {
            topoSort(root_);
            backwordTopo = forwordTopo;
            std::reverse(backwordTopo.begin(), backwordTopo.end());
            for (const shared_ptr<OpNode>& node: forwordTopo)
                node->setCuda(cuda_);
        }

        void topoSort(shared_ptr<OpNode>& root) {
            for (shared_ptr<OpNode> node: root->inNodes_)
                topoSort(node);
            forwordTopo.push_back(root);
        }

        tensorSp forward() {
            set<shared_ptr<OpNode>> s;
            for (shared_ptr<OpNode>& node: forwordTopo) {
                if (s.find(node) == s.end()) {
                    node->forward();
                    s.insert(node);
                }
            }

            return root_->getValue();
        }

        void backward(tensorSp grad, bool isTrain) {
            backwordTopo[0]->setGrad(std::move(grad));
            for (const auto& node: backwordTopo) {
                auto nodeGrads = node->backward();
                for (int i = 0; i < nodeGrads.size(); ++i) {
                    auto inputGrad = node->getInputNodes()[i]->getGrad();
                    cublasSaxpy(cuda_->cublas_,
                                inputGrad->size(),
                                &cuda_->one,
                                nodeGrads[i]->gpu(), 1,
                                inputGrad->gpu(), 1);
                }
                if (isTrain) node->updateValue(lr_);
            }
        }

        void zeroGrad() {
            set<shared_ptr<OpNode>> s;
            for (shared_ptr<OpNode>& node: forwordTopo) {
                if (s.find(node) == s.end()) {
                    node->zeroGrad();
                    s.insert(node);
                }
            }
        }

        void setLr(float lr) {
            lr_ = lr;
        }

        void print() {
            set<shared_ptr<OpNode>> s;
            for (shared_ptr<OpNode>& node: forwordTopo) {
                if (s.find(node) == s.end()) {
                    node->print();
                    s.insert(node);
                }
            }
        }
    };

    typedef std::shared_ptr<OpNode> nodeSp;
    typedef std::shared_ptr<Identity> idenSp;
}

#endif //CUDA_CNN_AUTOGRAD_H
