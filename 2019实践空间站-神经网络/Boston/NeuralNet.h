
#include"Reader.h"

class params                    //参数类
{
public:
    int sample, feature;       //样本数， 特征数
    double eta, eps;           //学习率， 目标损失
    int max_epoch, batch_size; //全样本最大迭代次数， 每批次训练数量
};

class NeuralNet: public Reader
{
private:
    params hp; 
                        
    VectorXd W;         // 特征值权重
    double B;           // 偏移量

    VectorXd forward(); //前向计算
    void backward(MatrixXd& batch_x, VectorXd& batch_y, VectorXd& batch_z);  // 反向传播

public:
    NeuralNet(params);
    void inference(VectorXd& x);
    void train();
};

NeuralNet::NeuralNet(params hp): Reader(hp.sample, hp.feature)
{
    this->hp = hp;
    
    W = VectorXd::Zero(feature);
    B = 0.0;
}

VectorXd NeuralNet::forward()
{
    VectorXd Z = Xtrain * W + VectorXd::Constant(Xtrain.rows(),B); 
    return Z;
}

void NeuralNet::backward(MatrixXd& batch_x, VectorXd& batch_y, VectorXd& batch_z)
{
    VectorXd dZ = batch_z - batch_y;

    double dB = dZ.sum() / (hp.batch_size);
    VectorXd dW = batch_x.transpose() * dZ / hp.batch_size;

    W -= dW * hp.eta;
    B -= dB * hp.eta;
}

void NeuralNet::train()
{
    int max_iteration = sample/hp.batch_size;

    for(int epoch=1; epoch <= hp.max_epoch; epoch++)
    {
        for(int iteration=1; iteration <= max_iteration; iteration++)
        {
            getbatchsample(hp.batch_size, iteration);
            VectorXd Z = forward();
            backward(Xtrain, Ytrain, Z);
        }
        shuffle();

        if(!(epoch%10)) 
        {
            getwholesample();
            VectorXd Z = forward();
            VectorXd dZ = Z - Ytrain;

            double loss = dZ.dot(dZ) / (2*sample);
            cout << "epoch "<< epoch <<": "<<"loss = "<< loss << endl;

            if(loss < hp.eps)
            {
                cout<<"\nBingo!\n"<<endl;
                break;
            }
        }
    }
    cout<< endl;
    cout<<"normalized W = " <<W.transpose()<<endl;
    cout<<"normalized B = " <<B<<endl;  
}

void NeuralNet::inference(VectorXd& x)
{
    cout << endl;
    cout << "given x = "<< x.transpose() << endl;

    for(int i=0; i<feature; i++) x(i) = (x(i)-Xmin(i))/(Xmax(i)-Xmin(i));
    cout << "normalized x = "<< x.transpose() << endl;

    double ans = x.dot(W) + B;
    cout << "normalized y = "<< ans << endl;

    ans = ans*(Ymax -Ymin) + Ymin;
    cout << "inference y = "<< ans << endl;
}