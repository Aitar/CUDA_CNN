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

    class MerticsLogger {
    private:
        int maxEpoch;
        int maxStep;
        int logFreq;
        float accSum_ = 0.f;
        float lossSum_ = 0.f;
        int n_ = 0;

        std::vector<float> losses_;
        std::vector<float> accs_;

    public:
        MerticsLogger(int maxEpoch, int maxStep, int logFreq) : maxEpoch(maxEpoch), maxStep(maxStep), logFreq(logFreq) {}

        void newEpoch() {
            losses_.push_back(lossSum_ / n_);
            accs_.push_back(accSum_ / n_);
            accSum_ = 0.f;
            lossSum_ = 0.f;
            n_ = 0;
            printf("[Info] Epoch avg loss: %7.4f, avg acc: %5.2f.\n", losses_[losses_.size() - 1], accs_[accs_.size() - 1]);
        }

        void log(int epoch, int step, float lossValue, std::shared_ptr<Tensor> predicts, std::shared_ptr<Tensor> labels) {
            n_++;
            accSum_ += getAccurary(std::move(predicts), std::move(labels));
            lossSum_ += lossValue;

            if (step % logFreq == 0) {
                lossValue = lossSum_ / n_;
                float acc = 100 * accSum_ / n_;
                printf("[Info] Train epoch [%3d/%3d], step [%4d/%4d], loss: %7.4f, acc: %5.2f.\n", maxEpoch, epoch,
                       maxStep, step, lossValue, acc);
            }
        }

        static float getAccurary(std::shared_ptr<Tensor> predicts, std::shared_ptr<Tensor> labels) {
            int maxIdx, label;
            float rightCnt = 0.f, maxLogit, logit;
            for (int b = 0; b < predicts->n(); ++b) {
                label = (int)labels->cpu()[b];
                maxIdx = -1;
                maxLogit = 0.f;
                for (int i = 0; i < predicts->w(); ++i) {
                    logit = predicts->getItem(b, 0, 0, i);
                    if (maxLogit < logit) {
                        maxLogit = logit;
                        maxIdx = i;
                    }
                }
                if (maxIdx == label) ++rightCnt;
            }
            return rightCnt / predicts->n();
        }
    };
}
#endif //MYNN_DATA_UTILS_H
