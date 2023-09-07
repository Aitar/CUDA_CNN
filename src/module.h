#ifndef MYNN_MODULE_H
#define MYNN_MODULE_H

#include <utility>

#include "Tensor.h"
#include "Layer.h"

namespace cuDL {
    class Module {

    protected:
        std::map<std::string, std::shared_ptr<Layer>> layers_;
        std::vector<std::pair<std::string, std::string>> edges_;
        std::vector<std::shared_ptr<Layer>> sequence_;
        std::shared_ptr<Softmax> loss_ = nullptr;
        std::shared_ptr<CudaContext> cuda_ = nullptr;
        std::shared_ptr<Executor> executor_ = nullptr;
        std::shared_ptr<Identity> input_ = nullptr;

        bool isInit_ = false;
        bool autoGrad_ = true;
        bool isTrain_;
        float lr_;

    public:
        void zeroGrad() {
            if (autoGrad_) {
                NIL(executor_)->zeroGrad();
            } else {
                for (std::shared_ptr<Layer> &layer: sequence_) {
                    NIL(layer)->zeroGrad();
                }
            }
        };


        void train() { isTrain_ = true; }

        void infer() { isTrain_ = false; }

        void setCuda(std::shared_ptr<CudaContext> cudaCtx) {
            if (!isInit_) {
                cuda_ = std::move(cudaCtx);
                for (auto& layer: layers_) {
                    layer.second->setCudaContext(cuda_);
                }
            }
        }

        void init(std::shared_ptr<Tensor> x) {
            isInit_ = true;
            input_ = make_shared<Identity>(std::move(x));
            for (auto &edge : edges_) {
                auto pre = edge.first;
                auto cur = edge.second;
                if (pre == "input")
                    getLayer(cur)->makeGraph(input_);
                else
                    getLayer(cur)->makeGraph(getLayer(pre)->outputNode_);
            }
            executor_ = make_shared<Executor>(getLayer(edges_[edges_.size()-1].second)->outputNode_, cuda_);
            executor_->setLr(lr_);
        }


        void addLayer(const std::string &key, std::shared_ptr <Layer> layer) {
            sequence_.push_back(layer);
            layers_[key] = std::move(layer);
        }


        std::shared_ptr<Layer> getLayer(const string& name) {
            return layers_[name];
        }

        void connect(const string& pre, const string& cur) {
            edges_.emplace_back(pre, cur);
        }


        virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> x) {
            if (autoGrad_) {
                if (!isInit_) init(x);
                input_->setValue(x);
                x = executor_->forward();
            } else {
                for (std::shared_ptr<Layer> &layer: sequence_)
                    x = layer->forward(x);
            }
            return x;
        }


        virtual void makeGraph() = 0;


        void setLr(float lr) {
            lr_ = lr;
        }


        virtual void backward(const std::shared_ptr <Tensor> &lossGrad) final {
            if (autoGrad_) {
                executor_->backward(lossGrad, isTrain_);
            } else {
                if (isTrain_) {
                    std::shared_ptr<Tensor> grad = nullptr;
                    for (auto it = sequence_.rbegin(); it != sequence_.rend(); ++it) {
                        grad = it->get()->backward(grad);
                        if (it->get()->hasParams_)
                            it->get()->updateParams(lr_);
                    }
                }
            }
        };

        void print() {
            executor_->print();
        }
    };
}

#endif //MYNN_MODULE_H
