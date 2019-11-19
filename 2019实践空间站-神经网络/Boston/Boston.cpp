
#include"NeuralNet.h"

int main(int argc, char** argv)
{
    params hp;
    hp.batch_size = 10;
    hp.eps = 1e-5;
    hp.eta = 0.015;
    hp.max_epoch = 800;
    hp.sample = 506;
    hp.feature = 13;

    NeuralNet net(hp);

    net.read();
    net.normalize(); 
    
    net.train();

//sample 344:
    VectorXd w(hp.feature);
    w<<0.02543,55.00,3.780,0,0.4840,6.6960,56.40,5.7321,5,370.0,17.60,396.90,7.18;

//real = 23.9   inferece = 27.7
    net.inference(w);
    getchar();

    return 0;
}

