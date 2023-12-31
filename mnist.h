#ifndef MYNN_MNIST_H
#define MYNN_MNIST_H

#include <utility>

#include "src/Tensor.h"
#include "src/module.h"
#include "src/Layer.h"
#include "src/data_utils.h"

namespace cuDL{

    class MNIST: public Dataset {
    public:
        std::vector<std::vector< float>> data_;
        std::string labelPath_;
        std::string dataPath_;

        MNIST(std::string dataPath, std::string labelPath) :
                labelPath_(std::move(labelPath)), dataPath_(std::move(dataPath)) {
            c_ = 1;
            loadData();
            loadLabels();
        }


        void loadLabels() {
            std::cout << "[Info] Loading \"" << labelPath_ << "\"...";
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
            printf("success.\n");
            file.close();
        }

        void loadData() {
            std::cout << "[Info] Loading \"" << dataPath_ << "\"...";
            uint8_t ptr[4];

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
            std::cout << "success." << std::endl;
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

    class Net: public Module {
    public:
        int nClass_;

        Net(int nClass) : nClass_(nClass) {
            loss_ = std::make_shared<Softmax>();
            addLayer("conv", std::make_shared<Conv2D>(1, 2, 3));
            addLayer("bn1", std::make_shared<BatchNorm>());
            addLayer("max_pool", std::make_shared<MaxPooling>(3));
            addLayer("relu1", std::make_shared<ReLU>());
            addLayer("relu2", std::make_shared<ReLU>());
            addLayer("fc", std::make_shared<Linear>(1152, nClass_));
            addLayer("bn2", std::make_shared<BatchNorm>());
            addLayer("softmax", loss_);
        }

        std::shared_ptr <Tensor> forward(std::shared_ptr<Tensor> input) override {
            return Module::forward(std::move(input));
        }

        void makeGraph() override {
            connect( "input", "conv");
            connect("conv", "bn1");
            connect("bn1", "relu1");
            connect("relu1", "max_pool");
            connect("max_pool", "fc");
            connect("fc", "relu2");
            connect("relu2", "bn2");
            connect("bn2", "softmax");
        }
    };

} // namespace cudl


#endif //MYNN_MNIST_H
