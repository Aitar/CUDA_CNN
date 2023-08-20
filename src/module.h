//
// Created by didi on 2023/8/20.
//

#ifndef MYNN_MODULE_H
#define MYNN_MODULE_H

#include "Tensor.h"
#include "Layer.h"

namespace cuDL {
    class Module {
    protected:
        std::map <std::string, std::shared_ptr<Layer>> layers_;
        std::vector <std::shared_ptr<Layer>> sequence_;
        std::shared_ptr <Softmax> loss_ = nullptr;
        std::shared_ptr <CudaContext> cudaCtx_;

        bool isTrain_{};
        float lr_{};
    public:
        void zeroGrad() {};

        void train() { isTrain_ = true; }

        void infer() { isTrain_ = false; }

        void init(std::shared_ptr <CudaContext> cudaCtx) {
            cudaCtx_ = cudaCtx;
            for (auto &layer: layers_) {
                layer.second->setCudaContext(cudaCtx_);
            }
        }

        void addLayer(const std::string &key, std::shared_ptr <Layer> layer) {
            sequence_.push_back(layer);
            layers_[key] = std::move(layer);
        }

        virtual std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> x) {
            for (std::shared_ptr <Layer> &layer: sequence_)
                x = layer->forward(x);
            return x;
        }

        void setLr(float lr) { lr_ = lr; }

        virtual float backward(const std::shared_ptr <Tensor> &labels) final {
            float lossValue = loss_->getLoss(labels);
            if (isTrain_) {
                std::shared_ptr<Tensor> grad = nullptr;
                for (auto it = sequence_.rbegin(); it != sequence_.rend(); ++it) {
                    grad = it->get()->backward(grad);
                    it->get()->updateParams(lr_);
                }
            }
            return lossValue;
        };
    };
}

#endif //MYNN_MODULE_H
