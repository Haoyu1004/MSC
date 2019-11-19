
#include"Reader.h"

class params                    //参数类
{
public:
    int sample, feature, tag;       //样本数， 特征数
    double eta, eps;           //学习率， 目标损失
    int max_epoch, batch_size; //全样本最大迭代次数， 每批次训练数量
};

class NeuralNet: public Reader
{
private:
    params hp; 
                        
    MatrixXd W;         // 特征值权重
    RowVectorXd B;      // 偏移量

    MatrixXd forward(); //前向计算
    
    void backward(MatrixXd& batch_x, MatrixXd& batch_y, MatrixXd& batch_z);  // 反向传播
    double checkloss();

public:
    NeuralNet(params);
    void inference(MatrixXd& x);
    void train();
    MatrixXd softmax(MatrixXd& Z);
};

NeuralNet::NeuralNet(params hp): Reader(hp.sample, hp.feature, hp.tag)
{
    this->hp = hp;
    
    W = MatrixXd::Zero(feature, tag);
    B = RowVectorXd::Zero(tag);
}

MatrixXd NeuralNet::softmax(MatrixXd& Z)
{
    MatrixXd A = Z;
    for(int i=0; i<A.rows(); i++)
    {
        double max = A.row(i).maxCoeff();
        for(int j=0; j<Z.cols(); j++) A(i,j) -= max;

        for(int j=0; j<Z.cols(); j++) A(i,j) = exp(A(i,j));
        
        double sum = A.row(i).sum();
        A.row(i) /= sum;
    }
    return A;
}

MatrixXd NeuralNet::forward()
{
    MatrixXd Z = Xtrain * W;
    for(int i=0; i<Z.rows(); i++) Z.row(i) += B;

    return Z;
}

void NeuralNet::backward(MatrixXd& batch_x, MatrixXd& batch_y, MatrixXd& batch_a)
{
    MatrixXd dZ = batch_a - batch_y;

    RowVectorXd dB(dZ.cols());
    for(int j=0; j<dZ.cols(); j++) dB(j) = dZ.col(j).sum() / hp.batch_size;

    MatrixXd dW = batch_x.transpose() * dZ / hp.batch_size;

    W -= dW * hp.eta;
    B -= dB * hp.eta;
}

double NeuralNet::checkloss()
{
    getwholesample();
    MatrixXd Z = forward();
    MatrixXd A = softmax(Z);

    double ans = 0.0;
    for(int i=0; i<A.rows(); i++)
        for(int j=0; j<A.cols(); j++)
            ans += Ytrain(i,j) * log(A(i,j));

    ans = -ans / sample;

    return ans;
}

void NeuralNet::train()
{
    int max_iteration = sample/hp.batch_size;

    for(int epoch=1; epoch <= hp.max_epoch; epoch++)
    {
        for(int iteration=1; iteration <= max_iteration; iteration++)
        {
            getbatchsample(hp.batch_size, iteration);
            MatrixXd Z = forward();
            MatrixXd A = softmax(Z);
            backward(Xtrain, Ytrain, A);
        }
        shuffle();

        if(!(epoch%10)) 
        {
            double loss = checkloss();
            cout << "epoch "<< epoch <<": "<<"loss = "<< loss << endl;

            if(loss < hp.eps)
            {
                cout<<"\nBingo!\n"<<endl;
                break;
            }
        }   
    }
    cout<< endl;
    cout<<"normalized W =\n" <<W<<"\n\n";
    cout<<"normalized B =\n" <<B<<"\n"; 


}

void NeuralNet::inference(MatrixXd& x)
{
    cout << endl;
    cout << "given x =\n"<< x <<"\n\n";

    for(int i=0; i<x.rows(); i++)
        for(int j=0; j<feature; j++) 
            x(i,j) = (x(i,j)-Xmin(j))/(Xmax(j)-Xmin(j));
    
    cout << "normalized x =\n"<< x <<"\n\n";

    Xtrain = x;
    MatrixXd Z = forward();
    MatrixXd A = softmax(Z);
    cout << "A =\n" << A <<"\n\n";

    VectorXd ans(x.rows());
    for(int i=0; i<x.rows(); i++)
    {
        int maxtag = 0;
        for(int j=0; j<tag; j++) if( A(i,j) > A(i,maxtag) ) maxtag = j;
        ans(i) = maxtag + 1;
    }
    cout << "r = \n" << ans <<"\n";

}
