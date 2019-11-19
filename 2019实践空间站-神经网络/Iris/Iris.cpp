
#include"NeuralNet.h"

int main(int argc, char** argv)
{
    params hp;
    hp.batch_size = 10;
    hp.eps = 0.1;
    hp.eta = 0.1;
    hp.max_epoch = 5000;
    hp.sample = 150;
    hp.feature = 4;
    hp.tag = 3;

    NeuralNet net(hp);

    net.read();
    net.normalize(); 

    net.train();

    int test_sample = 4;
    MatrixXd test(test_sample, hp.feature);

    test <<
    5.7,2.6,3.5,1.0, //sample 80, 2 Iris-versicolor
    4.9,3.1,1.5,0.1, //sample 35, 1 Iris-setosa
    5.8,2.7,5.1,1.9, //sample 102, 3 Iris-virginica
    5.0,2.0,3.5,1.0; //sample 61, 2 Iris-versicolor

    net.inference(test);

    return 0;
}

