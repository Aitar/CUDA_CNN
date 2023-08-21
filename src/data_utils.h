//
// Created by Aitar Hwan on 2023/3/16.
//
#ifndef MYNN_DATA_UTILS_H
#define MYNN_DATA_UTILS_H
# include <cassert>
# include "Tensor.h"

typedef enum {
    DISK, MEMORY
} DATA_LOC;

namespace cuDL {
    class Dataset {
    protected:
        DATA_LOC dataLoc_;
        std::vector<float> labels_;
    public:
        virtual void getItem(float *dataPtr, int index) = 0;

        virtual int getLen() = 0;

        int itemSize() { return c_ * h_ * w_; };

        float getLabel(int index) { return labels_[index]; }

        int c_, h_, w_;
    };

    class DataLoader {
    private:
        std::shared_ptr<Dataset> dataset_;
        std::shared_ptr<Tensor> data_;
        std::shared_ptr<Tensor> lables_;
        std::vector<int> idxs_;
        int batchSize_;

    public:
        DataLoader(std::shared_ptr<Dataset> dataset, int batchSize, bool shuffle = false) : batchSize_(batchSize) {
            dataset_ = std::move(dataset);
            for (int i = 0; i < dataset_->getLen(); ++i)
                idxs_.push_back(i);

            if (shuffle) {
                std::random_device rd;
                std::mt19937 rng(rd());
                std::shuffle(idxs_.begin(), idxs_.end(), rng);
            }
        }

        std::array<std::shared_ptr<Tensor>, 2> getData(int &step) {
            int offset = step * batchSize_;
            int n = std::min(batchSize_, dataset_->getLen() - offset);
            auto data = std::make_shared<Tensor>(batchSize_, dataset_->c_, dataset_->h_, dataset_->w_);
            auto labels = std::make_shared<Tensor>(batchSize_, 1, 1, 1);
            for (int i = 0; i < n; ++i) {
                int idx = idxs_[i];
                dataset_->getItem(data->cpu() + i * dataset_->itemSize(), idx);
                labels->cpu()[i] = dataset_->getLabel(idx);
            }
            std::array<std::shared_ptr<Tensor>, 2> res = {data, labels};
//            auto res = std::pair(std::move(data), std::move(labels));
            return res;
        }
    };
}
#endif //MYNN_DATA_UTILS_H
