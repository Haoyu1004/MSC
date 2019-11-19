
#include"Boston.h"

class Reader
{
private:
    MatrixXd Xraw, Xnorm;
    VectorXd Yraw, Ynorm;
    random_device rd;
protected:
    int sample, feature;//总样本数量，特征值数量
    
    MatrixXd Xtrain;
    VectorXd Ytrain;//该批次训练样本

    double Ymax, Ymin;
    VectorXd Xmax, Xmin;//用于反归一化

    void getbatchsample(int batch_size, int iteration);
    void getwholesample();//获取小批次/全批次训练样本
    Reader(int sample, int feature);

    void shuffle();

public:
    void read();//数据读入
    void normalize();// 样本&标签的归一化
};

Reader::Reader(int sample, int feature)
{   
    this->sample = sample;
    this->feature = feature;
    Xraw.resize(sample, feature);
    Yraw.resize(sample);
}

void Reader::read()
{
    ifstream inputs;
    inputs.open("housing.data",ios::in);
    
    for(int i=0; i<sample; i++)
    {
        for(int j=0; j<feature; j++) inputs >> Xraw(i,j);
        inputs >> Yraw(i);
    }
    inputs.close();

    Xnorm = Xraw;
    Ynorm = Yraw;
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

    {
        Ymax = Ynorm.maxCoeff();
        Ymin = Ynorm.minCoeff();

        VectorXd Min(sample);
        Min = VectorXd::Constant(sample, Ymin);

        Ynorm -= Min;
        Ynorm /= (Ymax-Ymin);
    }
    //cout<<Xnorm.row(500);
}


void Reader::getbatchsample(int batch_size, int iteration)
{
    Xtrain = Xnorm.block((iteration-1)*batch_size, 0, batch_size, feature);
    Ytrain = Ynorm.block((iteration-1)*batch_size, 0, batch_size, 1);
}

void Reader::getwholesample()
{
    Xtrain = Xnorm;
    Ytrain = Ynorm;
}

void Reader::shuffle()
{
    unsigned seed = rd()+time(NULL); //rd每个循环随机数不同，time每次运行随机数不同

    int v[sample];
    for(int i=0; i<sample; i++) v[i]=i;

    std::shuffle(v,v+sample,default_random_engine(seed));

    MatrixXd Xnew(sample, feature);
    VectorXd Ynew(sample);

    for(int i=0; i<sample; i++)
    {
        Xnew.row(i) = Xnorm.row(v[i]);
        Ynew(i) = Ynorm(v[i]);
    }

    Xnorm = Xnew;
    Ynorm = Ynew;

}