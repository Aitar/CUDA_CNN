#include <iostream>
#include <queue>
#include "mnist.h"
#include "src/utils.cuh"

using namespace std;
using namespace cuDL;


int main() {
    const string trainDataPath = "/tmp/tmp.wZ4VFMY5rG/data/train-images-idx3-ubyte";
    const string trainLabelPath = "/tmp/tmp.wZ4VFMY5rG/data/train-labels-idx1-ubyte";
    const string testDataPath = "/tmp/tmp.wZ4VFMY5rG/data/t10k-images-idx3-ubyte";
    const string testLabelPath = "/tmp/tmp.wZ4VFMY5rG/data/t10k-labels-idx1-ubyte";

    const int batchSize = 64;
    const int maxEpoch = 10;
    const int testFrequency = 10;
    const int printFrequency = 1;
    float lr = 0.02;

    auto cuda = make_shared<CudaContext>();
    auto trainDataset = make_shared<MNIST>(trainDataPath, trainLabelPath);
    auto testDataset = make_shared<MNIST>(testDataPath, testLabelPath);
    auto trainDataLoader = new DataLoader(trainDataset, batchSize, true);
    auto testDataLoader = new DataLoader(testDataset, batchSize, true);
    auto model = new Net(10);

    auto loss = make_shared<CrossEntropy>(cuda);
    model->setLr(lr);
    model->setCuda(cuda);
    model->makeGraph();

    int trainSteps = trainDataset->getLen() / batchSize + 1;
    int testSteps = testDataset->getLen() / batchSize + 1;

    auto trainLogger = new MerticsLogger(maxEpoch, trainSteps, printFrequency);
    auto testLogger = new MerticsLogger(maxEpoch, testSteps, printFrequency);

    for (int epoch = 0; epoch < maxEpoch; ++epoch) {
        model->train();
        for (int step = 0; step < trainSteps; ++step) {
            auto pair = trainDataLoader->getData(step);
            auto onehotLabels = onehot(pair[1], 10);
            auto logit = model->forward(pair[0]);
            auto lossValue = loss->getLoss(logit, onehotLabels);
            model->backward(loss->backward());
            model->zeroGrad();
            lossValue->print();
            trainLogger->log(epoch, step, lossValue->cpu()[0], logit, pair[1]);
        }

        if (epoch % testFrequency == 0) {
            model->infer();
            for (int step = 0; step < testSteps; ++step) {
                auto pair = testDataLoader->getData(step);
                auto logit = model->forward(pair[0]);
                auto lossValue = loss->getLoss(logit, pair[1]);
                testLogger->log(epoch, step, lossValue->cpu()[0], logit, pair[1]);
            }
        }
        trainLogger->newEpoch();
    }

    return 0;
}
