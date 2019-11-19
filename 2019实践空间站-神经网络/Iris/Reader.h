
#include"Iris.h"

class Reader
{
private:
    MatrixXd Xraw, Xnorm;
    MatrixXd Y;
    random_device rd;
protected:
    int sample, feature, tag;//总样本数量，特征值数量
    
    MatrixXd Xtrain;
    MatrixXd Ytrain;//该批次训练样本

    VectorXd Xmax, Xmin;//用于反归一化

    void getbatchsample(int batch_size, int iteration);
    void getwholesample();//获取小批次/全批次训练样本
    Reader(int sample, int feature, int tag);

    void shuffle();

public:
    void read();//数据读入
    void normalize();// 样本&标签的归一化
};

Reader::Reader(int sample, int feature, int tag)
{   
    this->sample = sample;
    this->feature = feature;
    this->tag = tag;
    Xraw.resize(sample, feature);
    Y.resize(sample, tag);
}

void Reader::read()
{
    ifstream inputs;
    inputs.open("iris.data",ios::in);
    
    for(int i=0; i<sample; i++)
    {
        //4.6,3.1,1.5,0.2,Iris-setosa
        char ch, s[20];
        for(int j=0; j<feature; j++) inputs >> Xraw(i,j) >> ch;

        inputs >> s;
        if(!strcmp(s,"Iris-setosa")) Y.row(i) << 1, 0, 0;
        else if(!strcmp(s, "Iris-versicolor")) Y.row(i) << 0, 1, 0;
        else if(!strcmp(s, "Iris-virginica")) Y.row(i) << 0, 0, 1;
        else Y.row(i) << 1, 1, 1;

    }
    inputs.close();

    Xnorm = Xraw;
}

void Reader::normalize()
{
    Xmax.resize(feature);
    Xmin.resize(feature);
    for(int j=0; j<feature; j++)
    {
        Xmax(j) = Xnorm.col(j).maxCoeff();
        Xmin(j) = Xnorm.col(j).minCoeff();

        VectorXd Min(sample); 
        Min = VectorXd::Constant(sample, Xmin(j));
        
        Xnorm.col(j) -= Min;
        Xnorm.col(j) /= (Xmax(j)-Xmin(j));
    }
}


void Reader::getbatchsample(int batch_size, int iteration)
{
    Xtrain = Xnorm.block((iteration-1)*batch_size, 0, batch_size, feature);
    Ytrain = Y.block((iteration-1)*batch_size, 0, batch_size, tag);
}

void Reader::getwholesample()
{
    Xtrain = Xnorm;
    Ytrain = Y;
}

void Reader::shuffle()
{
    unsigned seed = rd()+time(NULL); //rd每个循环随机数不同，time每次运行随机数不同

    int v[sample];
    for(int i=0; i<sample; i++) v[i]=i;

    std::shuffle(v,v+sample,default_random_engine(seed));

    MatrixXd Xnew(sample, feature);
    MatrixXd Ynew(sample, tag);

    for(int i=0; i<sample; i++)
    {
        Xnew.row(i) = Xnorm.row(v[i]);
        Ynew.row(i) = Y.row(v[i]);
    }

    Xnorm = Xnew;
    Y = Ynew;

}