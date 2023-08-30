#ifndef MYNN_MODULE_H
#define MYNN_MODULE_H

#include <utility>

#include "Tensor.h"
#include "Layer.h"

namespace cuDL {
    class Module {

    protected:

        std::map<std::string, std::shared_ptr<Layer>> layers_;
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
            for (std::shared_ptr <Layer> &layer: sequence_) {
                layer->zeroGrad();
            }
        };


        void train() { isTrain_ = true; }


        void infer() { isTrain_ = false; }


        void init(std::shared_ptr <CudaContext> cudaCtx) {
            if (!isInit_) {
                cuda_ = std::move(cudaCtx);
                for (auto& layer: layers_) {
                    layer.second->setCudaContext(cuda_);
                }
            }
        }


        void addLayer(const std::string &key, std::shared_ptr <Layer> layer) {
            sequence_.push_back(layer);
            layers_[key] = std::move(layer);
        }


        std::shared_ptr<Layer> getLayer(const string& name) {
            return layers_[name];
        }


        virtual std::shared_ptr<Tensor> forward(std::shared_ptr <Tensor> x) {
            if (autoGrad_) {
                if (input_ == nullptr) makeGraph(x);
                executor_->forward();
            } else {
                for (std::shared_ptr <Layer> &layer: sequence_)
                    x = layer->forward(x);
            }
            return x;
        }


        virtual void makeGraph(std::shared_ptr<Tensor> input) {
            input_ = make_shared<Identity>(std::move(input));
            sequence_[0]->makeGraph(input_);
            for (int i = 1; i < sequence_.size(); ++i)
                sequence_[i]->makeGraph(sequence_[i - 1]->getOutputNode());
            executor_ = make_shared<Executor>(sequence_[sequence_.size() - 1]->getOutputNode(), cuda_);
        }


        void setLr(float lr) { lr_ = lr; }


        virtual float backward(const std::shared_ptr <Tensor> &labels) final {
            float lossValue = loss_->getLoss(labels);
            if (isTrain_) {
                std::shared_ptr<Tensor> grad = nullptr;
                for (auto it = sequence_.rbegin(); it != sequence_.rend(); ++it) {
                    grad = it->get()->backward(grad);
                    if (it->get()->hasParams_)
                        it->get()->updateParams(lr_);
                }
            }
            return lossValue;
        };


        void print() {
            for (const auto& layer: sequence_)
                layer->print();
        }
    };
}

#endif //MYNN_MODULE_H
