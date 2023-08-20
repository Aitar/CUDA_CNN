#include <iostream>
#include "mnist.h"

using namespace std;
using namespace cuDL;

int main() {
    const string trainDataPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/train-images-idx3-ubyte";
    const string trainLabelPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/train-labels-idx1-ubyte";
    const string testDataPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/t10k-images-idx3-ubyte";
    const string testLabelPath = "/nfs/volume-73-1/huangdingli/my_workspace/cuda/myNN/data/t10k-labels-idx1-ubyte";

    const int batchSize = 10;
    const int maxEpoch = 100;
    const int testFrequency = 10;

    auto trainDataset = make_shared<MNIST>(trainDataPath, trainLabelPath);
    auto testDataset = make_shared<MNIST>(testDataPath, testLabelPath);
    auto trainDataLoader = new DataLoader(trainDataset, batchSize, true);
    auto testDataLoader = new DataLoader(testDataset, batchSize, true);
    auto model = new Net(10);

    int trainSteps = trainDataset->getLen() / batchSize + 1;
    int testSteps = testDataset->getLen() / batchSize + 1;

    for (int epoch = 0; epoch < maxEpoch; ++epoch) {
        for (int step = 0; step < trainSteps; ++step) {
            auto pair = trainDataLoader->getData(step);
            auto pre = model->forward(pair.first);
            float lossValue = model->backward(pair.second);
            printf("[Info] Train epoch [%8d], step [%8d], loss [%8.4f], acc[%8.4f].\n"
                   , epoch, step, lossValue, 0.0);
        }

        if (epoch % testFrequency == 0) {
            for (int step = 0; step < testSteps; ++step) {
                auto pair = testDataLoader->getData(step);
                auto pre = model->forward(pair.first);
                float lossValue = model->backward(pair.second);
                printf("[Info] Test epoch [%8d], step [%8d], loss [%8.4f], acc[%8.4f].\n"
                       , epoch, step, lossValue, 0.0);
            }
        }
    }

    return 0;
}
