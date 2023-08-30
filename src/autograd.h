#ifndef CUDA_CNN_AUTOGRAD_H
#define CUDA_CNN_AUTOGRAD_H

# include <string>
# include <utility>
# include <set>
# include "utils.cuh"

namespace cuDL{
    using namespace std;
    static int globalNodeID = 0;

    class OpNode{
    protected:
        tensorSp value_ = nullptr;
        tensorSp grad_ = nullptr;
        shared_ptr<CudaContext> cuda_ = nullptr;

        tensors inputGrads_;
        tensors params_;

        bool isInit = false;

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

        void setValue(tensorSp value) {
            value_ = std::move(value);
            if (grad_ == nullptr || grad_->size() != value_->size()) {
                grad_ = make_shared<Tensor>(value_, 1, false);
                cout << "[Waring] Value shape has changed, grad re-malloced." << endl;
            } else grad_->valueInit();
        }

        tensorSp getGrad() {
            return grad_;
        }

        void setGrad(tensorSp grad) {
            grad_ = std::move(grad);
        }


        void setCuda(shared_ptr<CudaContext> cuda) {
            cuda_ = std::move(cuda);
        }

        vector<shared_ptr<OpNode>>& getInputNodes() {
            return inNodes_;
        }

        void print() {
            cout << name_ << endl << "value:" << endl;
            value_->print();
            cout << "grad:" << endl;
            grad_->print();
        }
    };

    typedef vector<shared_ptr<OpNode>> nodes;


    class Add: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> b_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        shared_ptr<Tensor> db_ = nullptr;
        float alpha_;
        float beta_;

    public:
        Add(vector<shared_ptr<OpNode>> inNodes, float alpha = 1.f, float beta = 1.f) :
                 alpha_(alpha), beta_(beta) {
            setName("Add");
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            b_ = ckNullptr(inNodes_[1]->getValue());
            da_ = make_shared<Tensor>(a_->n(), 1, 1, a_->len());
            db_ = make_shared<Tensor>(b_->n(), 1, 1, b_->len());
            grad_ = make_shared<Tensor>(a_->n(), 1, 1, a_->len());
            inputGrads_.push_back(da_);
            inputGrads_.push_back(db_);
            value_ = make_shared<Tensor>(a_->n(), 1, 1, a_->len());
        }

        void forward() override {
            if (!isInit) init();
            CUBLASCHECK(cublasSgeam(ckNullptr(cuda_)->cublas_,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    a_->len(), a_->n(),
                                    &alpha_, a_->gpu(), a_->len(),
                                    &beta_, b_->gpu(), a_->len(),
                                    value_->gpu(), a_->n()));

        }

        tensors& backward() override {
            da_->copy(grad_);
            db_->copy(grad_);
            CUBLASCHECK(cublasSscal_v2(cuda_->cublas_, da_->size(), &alpha_,  da_->gpu(), 1));
            CUBLASCHECK(cublasSscal_v2(cuda_->cublas_, db_->size(), &beta_,  db_->gpu(), 1));

            return inputGrads_;
        }
    };

    class Gemm: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> b_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        shared_ptr<Tensor> db_ = nullptr;

    public:
        Gemm(nodes inNodes) {
            setName("Gemm");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            b_ = ckNullptr(inNodes_[1]->getValue());
            da_ = make_shared<Tensor>(a_->n(), 1, 1, a_->len());
            db_ = make_shared<Tensor>(b_->n(), 1, 1, b_->w());
            inputGrads_.push_back(da_);
            inputGrads_.push_back(db_);
            value_ = make_shared<Tensor>(a_->n(), 1, 1, b_->w());
            grad_ = make_shared<Tensor>(a_->n(), 1, 1, b_->w());
        }

        void forward() override {
            if (!isInit) init();
            CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                       CUBLAS_OP_N, CUBLAS_OP_N,
                                       b_->w(), a_->n(), a_->w(),
                                       &cuda_->one,
                                       b_->gpu(), b_->w(),
                                       a_->gpu(), a_->w(),
                                       &cuda_->zero,
                                       value_->gpu(), b_->w()));
        }

        tensors& backward() override {
            int m = a_->n(), k = a_->w(), n = b_->w();

            CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                       CUBLAS_OP_T, CUBLAS_OP_N,
                                       k, m, n,
                                       &cuda_->one,
                                       b_->gpu(), k,
                                       grad_->gpu(), n,
                                       &cuda_->zero,
                                       da_->gpu(), k));

            CUBLASCHECK(cublasSgemm_v2(cuda_->cublas_,
                                       CUBLAS_OP_N, CUBLAS_OP_T,
                                       n, k, m,
                                       &cuda_->one,
                                       grad_->gpu(), n,
                                       a_->gpu(), k,
                                       &cuda_->zero,
                                       db_->gpu(), n));

            return inputGrads_;
        }
    };

    class Mul: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> b_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        shared_ptr<Tensor> db_ = nullptr;
    public:
        Mul(nodes inNodes) {
            setName("Mul");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            b_ = ckNullptr(inNodes_[1]->getValue());
            da_ = make_shared<Tensor>(a_->n(), 1, 1, a_->len());
            db_ = make_shared<Tensor>(b_->n(), 1, 1, b_->len());
            inputGrads_.push_back(da_);
            inputGrads_.push_back(db_);
            value_ = make_shared<Tensor>(a_);
            grad_ = make_shared<Tensor>(a_, 1, false);
        }

        void forward() override {
            if (!isInit) init();
            hadamardPorduct(cuda_->one, a_->size(), a_->gpu(), b_->gpu(), value_->gpu());
        }

        tensors& backward() override {
            hadamardPorduct(cuda_->one, grad_->size(), grad_->gpu(), a_->gpu(), da_->gpu());
            hadamardPorduct(cuda_->one, grad_->size(), grad_->gpu(), b_->gpu(), db_->gpu());

            return inputGrads_;
        }
    };

    class Sum: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        float alpha_ = 1.f;
        float bias_ = 0.f;
        float p_ = 1.f;

    public:
        Sum(nodes inNodes) {
            setName("Sum");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        Sum(nodes inNodes, float alpha, float bias = 0.f, float p = 1.f)
        : alpha_(alpha), bias_(bias), p_(p) {
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            value_ = make_shared<Tensor>(1, 1, 1, 1);
            grad_ = make_shared<Tensor>(1, 1, 1, 1);
            da_ = make_shared<Tensor>(a_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
            vectorSum(a_->gpu(), value_->gpu(), a_->size(), alpha_, bias_, p_);
        }

        tensors& backward() override {
            float a = grad_->cpu()[0] * alpha_ * p_;
            CUBLASCHECK(cublasSscal_v2(cuda_->cublas_, da_->size(), &a,  da_->gpu(), 1));
            a *= -bias_;
            vectorValueAdd(cuda_->one, da_->size(), da_->gpu(), a);
            return inputGrads_;
        }
    };

    class Exp: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;

    public:
        Exp(nodes inNodes) {
                setName("Exp");
                
                inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            da_ = make_shared<Tensor>(a_, 1, false);
            grad_ = make_shared<Tensor>(a_, 1, false);
            value_ = make_shared<Tensor>(a_, 1, false);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
            if (a_->size() < 8 && a_->device() == CPU) {
                for (int i = 0; i < a_->size(); ++i)
                    value_->cpu()[i] = std::exp(a_->cpu()[i]);
            } else {
                exp(a_->gpu(), value_->gpu(), a_->size());
            }
        }

        tensors& backward() override {
            hadamardPorduct(1, grad_->size(), value_->gpu(), grad_->gpu(), da_->gpu());
            return inputGrads_;
        }
    };

    class Ln: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;

    public:
        Ln(nodes inNodes) {
            setName("Ln");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            da_ = make_shared<Tensor>(a_, 1, false);
            grad_ = make_shared<Tensor>(a_, 1, false);
            value_ = make_shared<Tensor>(a_, 1, false);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
            if (a_->size() < 8 && a_->device() == CPU) {
                for (int i = 0; i < a_->size(); ++i)
                    value_->cpu()[i] = std::log(a_->cpu()[i]);
            } else {
                log(a_->gpu(), value_->gpu(), a_->size());
            }
        }

        tensors& backward() override {
            pow(value_->gpu(), da_->gpu(), -1, value_->size());
            hadamardPorduct(1, da_->size(), da_->gpu(), grad_->gpu(), da_->gpu());
            return inputGrads_;
        }
    };

    class Expand: public OpNode {
    private:
        int dup_;
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;

    public:
        Expand(nodes inNodes, int dup): dup_(dup) {
            setName("Expand");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            da_ = make_shared<Tensor>(1, 1, 1, 1);
            grad_ = make_shared<Tensor>(1, 1, 1, dup_);
            value_ = make_shared<Tensor>(a_->n(), a_->c() * dup_, a_->h(), a_->w());
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
            value_->copy(a_, dup_);
        }

        tensors& backward() override {
            da_->cpu()[0] = grad_->cpu()[0];
            return inputGrads_;
        }
    };

    class Pow: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        float p_;

    public:
        Pow(nodes inNodes, float p) {
            setName("Pow");
            p_ = p;
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            isInit = true;
            a_ = ckNullptr(inNodes_[0]->getValue());
            da_ = make_shared<Tensor>(a_, 1, false);
            grad_ = make_shared<Tensor>(a_, 1, false);
            value_ = make_shared<Tensor>(a_, 1, false);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
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
            grad_ = make_shared<Tensor>(value_->n(), value_->c(), value_->h(), value_->w());
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

        void update(float& minusLr) {
            cublasSaxpy(cuda_->cublas_,
                        value_->size(),
                        &minusLr,
                        grad_->gpu(), 1,
                        value_->gpu(), 1);
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
             size_t workspaceSize,
             shared_ptr<CudaContext> cuda):

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
        }

        void init() {
            isInit = true;

            x_ = ckNullptr(inNodes_[0]->getValue());
            w_ = ckNullptr(inNodes_[1]->getValue());
            dx_ = make_shared<Tensor>(x_, 1, false);
            dw_ = make_shared<Tensor>(w_, 1, false);
            value_ = make_shared<Tensor>(x_->n(), w_->c(), outputH_, outputW_);
            grad_ = make_shared<Tensor>(x_->n(), w_->c(), outputH_, outputW_);

            inputGrads_.push_back(dx_);
            inputGrads_.push_back(dw_);
        }

        void forward() override {
            if (!isInit) init();
            CUDNNCHECK(cudnnConvolutionForward(cuda_->cudnn_,
                                               &cuda_->one, x_->desc_, x_->gpu(),
                                               filterDesc_, w_->gpu(),
                                               convDesc_, fwdAlgo_,
                                               workspace_, workspaceSize_,
                                               &cuda_->zero, value_->desc_, value_->gpu()));
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
        }
    };

    class Pooling: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;

        int outputH_;
        int outputW_;
        cudnnPoolingDescriptor_t poolingDesc_;

    public:
        Pooling(nodes inNodes,
                int outputH,
                int outputW,
                cudnnPoolingDescriptor_t poolingDesc,
                shared_ptr<CudaContext> cuda):

            outputH_(outputH),
            outputW_(outputW),
            poolingDesc_(poolingDesc) {
            setName("Pooling");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        Pooling(nodes inNodes,
                shared_ptr<Tensor> a,
                shared_ptr<Tensor> da,
                shared_ptr<Tensor> value,
                cudnnPoolingDescriptor_t poolingDesc,
                shared_ptr<CudaContext> cuda):
                a_(std::move(a)),
                da_(std::move(da)),
                poolingDesc_(poolingDesc) {

            setName("Pooling");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
            value_ = std::move(value);
        }

        void init() {
            isInit = true;
            if (a_ == nullptr) a_ = ckNullptr(inNodes_[0]->getValue());
            if (da_ == nullptr) da_ = make_shared<Tensor>(a_, 1, false);
            if (grad_ == nullptr) grad_ = make_shared<Tensor>(a_->n(), a_->c(), outputH_, outputW_);
            if (value_ == nullptr) value_ = make_shared<Tensor>(a_->n(), a_->c(), outputH_, outputW_);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
            CUDNNCHECK(cudnnPoolingForward(cuda_->cudnn_, poolingDesc_,
                                           &cuda_->one, a_->desc_, a_->gpu(),
                                           &cuda_->zero, value_->desc_, value_->gpu()));
        }

        tensors& backward() override {
            CUDNNCHECK(cudnnPoolingBackward(cuda_->cudnn_, poolingDesc_,
                                            &cuda_->one,
                                            value_->desc_, value_->gpu(),
                                            grad_->desc_, grad_->gpu(),
                                            a_->desc_, a_->gpu(),
                                            &cuda_->zero,
                                            da_->desc_, da_->gpu()));
            return inputGrads_;
        }
    };

    class Active: public OpNode {
    private:
        shared_ptr<Tensor> a_ = nullptr;
        shared_ptr<Tensor> da_ = nullptr;
        cudnnActivationDescriptor_t actDesc_;
        float coef_;

    public:
        Active(nodes inNodes,
               shared_ptr<Tensor> a,
               shared_ptr <Tensor> da,
               shared_ptr <Tensor> value,
               shared_ptr<Tensor> grad,
               cudnnActivationDescriptor_t actDesc,
               float coef,
               shared_ptr<CudaContext> cuda):

            a_(std::move(a)), da_(std::move(da)), actDesc_(actDesc), coef_(coef) {
            grad_ = std::move(grad);
            value_ = std::move(value);
            setName("Active");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        Active(nodes inNodes, cudnnActivationDescriptor_t actDesc, float coef) {
            coef_ = coef;
            actDesc_ = actDesc;
            setName("Active");
            
            inNodes_.insert(inNodes_.end(), inNodes.begin(), inNodes.end());
        }

        void init() {
            if (a_ == nullptr) a_ = ckNullptr(inNodes_[0]->getValue());
            if (da_ == nullptr) da_ = make_shared<Tensor>(a_, 1, false);
            if (grad_ == nullptr) da_ = make_shared<Tensor>(a_, 1, false);
            if (value_ == nullptr) da_ = make_shared<Tensor>(a_, 1, false);
            inputGrads_.push_back(da_);
        }

        void forward() override {
            if (!isInit) init();
            cudnnActivationForward(cuda_->cudnn_,
                                   actDesc_,
                                   &cuda_->one, a_->desc_, a_->gpu(),
                                   &cuda_->one, value_->desc_, value_->gpu());
        }

        tensors& backward() override {
            cudnnActivationBackward(cuda_->cudnn_,
                                    actDesc_,
                                    &cuda_->one,
                                    value_->desc_, value_->gpu(),
                                    grad_->desc_, grad_->gpu(),
                                    a_->desc_, a_->gpu(),
                                    &cuda_->one,
                                    da_->desc_, da_->gpu());
            return inputGrads_;
        }
    };

    class Executor {
    private:
        shared_ptr<OpNode> root_;
        shared_ptr<CudaContext> cuda_;
        nodes backwordTopo;
        nodes forwordTopo;

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

        void backward(tensorSp grad) {
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
            }
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
