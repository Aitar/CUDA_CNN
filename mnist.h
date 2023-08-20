//
// Created by didi on 2023/8/20.
//

#ifndef MYNN_MNIST_H
#define MYNN_MNIST_H

#include "src/Tensor.h"
#include "src/module.h"
#include "src/Layer.h"
#include "src/data_utils.h"

namespace cuDL{

    class MNIST : public Dataset {
    public:
        std::vector<std::vector< float>> data_;
        std::string labelPath_;
        std::string dataPath_;

        MNIST(const std::string &labelPath, const std::string &dataPath) :
                labelPath_(labelPath), dataPath_(dataPath) {
            c_ = 1;
            loadLabels();
            loadData();
        }

        void loadLabels() {
            uint8_t ptr[4];
            std::ifstream file(labelPath_.c_str(), std::ios::in | std::ios::binary);

            if (!file.is_open()) {
                printf("[Error] Open \"%s\" failed, check file path.\n", labelPath_.c_str());
                exit(-1);
            }

            file.read((char *) ptr, 4);
            int magic = toInt(ptr);
            assert((magic & 0xFFF) == 0x801);

            file.read((char *) ptr, 4);
            int nLabels = toInt(ptr);

            for (int i = 0; i < nLabels; i++) {
                file.read((char *) ptr, 1);
                labels_.push_back(static_cast<float>(ptr[0]));
            }

            file.close();
        }

        void loadData() {
            int num_steps_ = 0;

            uint8_t ptr[4];

            printf("loading %s", dataPath_.c_str());
            std::ifstream file(dataPath_.c_str(), std::ios::in | std::ios::binary);
            if (!file.is_open()) {
                printf("[Error] Open file failed, check file path.\n");
                exit(-1);
            }

            file.read((char *) ptr, 4);
            int magic = toInt(ptr);
            assert((magic & 0xFFF) == 0x803);


            file.read((char *) ptr, 4);
            int num = toInt(ptr);

            file.read((char *) ptr, 4);
            h_ = toInt(ptr);

            file.read((char *) ptr, 4);
            w_ = toInt(ptr);

            auto *q = new uint8_t[itemSize()];
            for (int i = 0; i < num; i++) {
                std::vector<float> image = std::vector<float>(itemSize());
                float *imgPtr = image.data();

                file.read((char *) q, itemSize());
                for (int j = 0; j < itemSize(); j++) {
                    imgPtr[j] = (float) q[j] / 255.f;
                }

                data_.push_back(image);
            }

            delete[] q;

            file.close();
        }

        void getItem(float *dataPtr, int index) override {
            for (int i = 0; i < itemSize(); ++i) {
                dataPtr[i] = data_[index][i];
            }
        }

        int getLen() override {
            return labels_.size();
        }
    };

    class Net : public Module {
    public:
        int nClass_;

        Net(int nClass) : nClass_(nClass) {
            loss_ = std::make_shared<Softmax>();
            addLayer("conv1", std::make_shared<Conv2D>(1, 10, 5));
            addLayer("max_pool1", std::make_shared<MaxPooling>(2));
            addLayer("relu1", std::make_shared<ReLU>());
            addLayer("conv2", std::make_shared<Conv2D>(10, 20, 5));
            addLayer("max_pool2", std::make_shared<MaxPooling>(2));
            addLayer("relu2", std::make_shared<ReLU>());
            addLayer("fc1", std::make_shared<Linear>(320, 50));
            addLayer("relu3", std::make_shared<ReLU>());
            addLayer("fc2", std::make_shared<Linear>(50, nClass));
            addLayer("softmax", loss_);
        }

        std::shared_ptr <Tensor> forward(std::shared_ptr <Tensor> input) override {
            return Module::forward(input);
        }
    };

} // namespace cudl


#endif //MYNN_MNIST_H
