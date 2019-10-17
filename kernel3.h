#ifndef RCINCDLTool3
#define RCINCDLTool3

#include <vector>
#include <string>
#include "dataTensor.h"

// Layers with weights. layerType == 6 (mod 7)
const int CONV1D = 76;
const int CONV2D = 6;
const int CONV3D = 20;
const int CONV1DBIAS = 90;
const int CONV2DBIAS = 27;
const int CONV3DBIAS = 34;
const int FC = 41;
const int FCDO = 48;
const int FCBIAS = 13;
const int FCBR = 55;
const int CBR2D = 62;
const int CBR3D = 69;
const int CBR1D = 83;

// Layers without output. layerType == 1 (mod 7)
const int DIFFSQUARE = 1;
const int NAIVELOSS = 15;
const int LOGSOFTMAX = 8;
const int EUCLIDIST = 22;

// Layers without input. layerType == 2 (mod 7)
const int INPUT = 2;

// Layers with multiple inputs. layerType == 3 (mod 7)
const int BINDNTOONE = 3;
const int SUMNTOONE = 10;
const int STRUCTBIND = 17; // Not operated in netStructTool yet.

// Layers with multiple outputs. layerType == 4 (mod 7)
const int SEPARATEONETON = 4;

// Other layers. layerType == 5 (mod 7)
const int MAXPOOLING2 = 68;
const int MAXPOOLING2X2 = 5;
const int MAXPOOLING2X2X2 = 12;
const int RELU = 19;
const int NORMALIZE = 26;
const int PROTO = 33;
const int MAXPOOLING2D = 40;
const int MAXPOOLING3D = 47;
const int ZEROPADDING1D = 75;
const int ZEROPADDING2D = 54;
const int ZEROPADDING3D = 61;

extern int ALLBREAK;


using namespace std;


__global__ void GPUVectorScalarAddUp(float* d_in, float* d_out, int dataL, float alpha);

class ProtoLayer{
    //protected:   
    
    public:
    int layerType, id;
    int BN, BNf, BNb, BNx, BNy, BNz, BNxf, BNyf, BNzf, BNxb, BNyb, BNzb, BNaddup, i, tempLength;
    dataTensor* input;
    dataTensor* output;
    dataTensor* dedx;
    dataTensor* dedy;
    dataTensor* dedw;
    dataTensor* alldedw;
    dataTensor* weight;
    int* d_paddingN;
    float dropOutRatio;
    ProtoLayer();
    int* filterDim;
    
    int inputDimN;
    int* inputDim;
    int outputDimN;
    int* outputDim;
    
    void setInputDim(int dimN, int* dim);
    void setOutputDim(int dimN, int* dim);
    void checkAndCorrectInputDim();
    void checkAndCorrectOutputDim();
    void checkAndCorrectdedxDim();
    void checkAndCorrectdedyDim();
    
    void setInput(dataTensor* in_x);
    void setOutput(dataTensor* in_y);
    void setdedx(dataTensor* in_dedx);
    void setdedy(dataTensor* in_dedy);
    virtual void setWeight(dataTensor* in_weight);
    void setDropOutRatio(float);
    
    virtual void setAll();
    
    virtual void forward();
    virtual void backward();
    virtual float forward(int n);
    virtual float forward(float * tagArray);
    virtual void backward(int n);
    virtual void adjust(float alpha, float batchSize);
    virtual dataTensor* get_dedw();
    
    virtual dataTensor* getWeight();
    virtual void loadWeight(int dataLength, float* in_weight);
    
    virtual void add_input(dataTensor* in_x);
    virtual void add_dedx(dataTensor* dedx);
    virtual void add_output(dataTensor* out_y);
    virtual void add_dedy(dataTensor* dedy);
    int getType();
};

class ReLULayer : public ProtoLayer{
    public:
    int dataLength;
    
    ReLULayer();
    void setAll();
    void forward();
    void backward();
};


class ZeroPad1DLayer : public ProtoLayer{
    public:
    int in_featureN;
    int out_featureN;
    int x_padding1, x_padding2;

    int outCount;
    int inCount;
    
    ZeroPad1DLayer();
    void setAll();
    void forward();
    void backward();
    void setPadding(int x1, int x2);
};


class ZeroPad2DLayer : public ProtoLayer{
    public:
    int in_featureN;
    int out_featureN;
    int x_padding1, x_padding2, y_padding1, y_padding2;

    int outCount;
    int inCount;
    
    ZeroPad2DLayer();
    void setAll();
    void forward();
    void backward();
    void setPadding(int x1, int x2, int y1, int y2);
};

class ZeroPad3DLayer : public ProtoLayer{
    public:
    int in_featureN;
    int out_featureN;
    int x_padding1, x_padding2, y_padding1, y_padding2, z_padding1, z_padding2;

    int outCount;
    int inCount;
    
    ZeroPad3DLayer();
    void setAll();
    void forward();
    void backward();
    void setPadding(int x1, int x2, int y1, int y2, int z1, int z2);
};

class Conv1DLayer : public ProtoLayer{
    public:
    int in_featureN;
    int out_featureN;
    int x_padding;

    dataTensor* buffer;// For each input feature map, used in forward() also in backward().
    int outCount;
    int inCount;
    
    Conv1DLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);
    void setWeight(int x);
};

class Conv2DLayer : public ProtoLayer{
    public:
    int in_featureN;
    int out_featureN;
    int x_padding, y_padding;

    dataTensor* buffer;// For each input feature map, used in forward() also in backward().
    int outCount;
    int inCount;
    
    Conv2DLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);
    void setWeight(int x, int y);
};

class Conv3DLayer : public ProtoLayer{
    public:
    int in_featureN;
    int out_featureN;
    int x_padding, y_padding, z_padding;
    dataTensor* buffer;// For each input feature map, used in forward() also in backward().
    int outCount;
    int inCount;
    
    Conv3DLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);
    void setWeight(int x, int y, int z);
};

class Conv1DBiasLayer : public ProtoLayer{
    public:
    int featureN;
    int n2;
    float* h_bias;
    
    Conv1DBiasLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);

};

class Conv2DBiasLayer : public ProtoLayer{
    public:
    int featureN;
    int n2;
    float* h_bias;
    
    Conv2DBiasLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);

};

class Conv3DBiasLayer : public ProtoLayer{
    public:
    int featureN;
    int n3;
    float* h_bias;
    
    Conv3DBiasLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);

};


class CBR1DLayer : public ProtoLayer{ // Conv1D + Bias + ReLU
    //private:
    public:
    Conv1DLayer* conv;
    Conv1DBiasLayer* bias;
    ReLULayer* relu;
    
    dataTensor* Conv2Bias;
    dataTensor* Bias2ReLU;
    dataTensor* Bias2Conv;
    dataTensor* ReLU2Bias;
    dataTensor* ConvWeight;
    dataTensor* BiasWeight;
    
    dataTensor* allWeight;

    //public:
    CBR1DLayer();
    void setAll();
    void forward();
    void backward();
    void setWeight(int x);
    
    void adjust(float alpha, float batchSize);
    
    dataTensor* getWeight();
    void loadWeight(int dataLength, float* in_weight);
};

class CBR2DLayer : public ProtoLayer{ // Conv2D + Bias + ReLU
    //private:
    public:
    Conv2DLayer* conv;
    Conv2DBiasLayer* bias;
    ReLULayer* relu;
    
    dataTensor* Conv2Bias;
    dataTensor* Bias2ReLU;
    dataTensor* Bias2Conv;
    dataTensor* ReLU2Bias;
    dataTensor* ConvWeight;
    dataTensor* BiasWeight;
    
    dataTensor* allWeight;

    //public:
    CBR2DLayer();
    void setAll();
    void forward();
    void backward();
    void setWeight(int x, int y);
    
    void adjust(float alpha, float batchSize);
    
    dataTensor* getWeight();
    void loadWeight(int dataLength, float* in_weight);
};

class CBR3DLayer : public ProtoLayer{ // Conv3D + Bias + ReLU
    private:
    Conv3DLayer* conv;
    Conv3DBiasLayer* bias;
    ReLULayer* relu;
    
    dataTensor* Conv2Bias;
    dataTensor* Bias2ReLU;
    dataTensor* Bias2Conv;
    dataTensor* ReLU2Bias;
    dataTensor* ConvWeight;
    dataTensor* BiasWeight;
    
    dataTensor* allWeight;

    public:
    CBR3DLayer();
    void setAll();
    void forward();
    void backward();
    void setWeight(int x, int y, int z);
    
    void adjust(float alpha, float batchSize);
    
    dataTensor* getWeight();
    void loadWeight(int dataLength, float* in_weight);
};

class MaxPooling2DLayer : public ProtoLayer{
    public:
    int nf, nb;
    int filterDim1, filterDim2, xPadding, yPadding;
    
    MaxPooling2DLayer();
    void setFilterDim(int inDim1, int inDim2);
    void setAll();
    void forward();
    void backward();
};

class MaxPooling3DLayer : public ProtoLayer{
    public:
    int nf, nb;
    int filterDim1, filterDim2, filterDim3, xPadding, yPadding, zPadding;
    
    MaxPooling3DLayer();
    void setFilterDim(int inDim1, int inDim2, int inDim3);
    void setAll();
    void forward();
    void backward();
};

class MaxPooling2x2x2Layer : public ProtoLayer{
    public:
    int nf, nb;
    
    MaxPooling2x2x2Layer();
    void setAll();
    void forward();
    void backward();
};


class MaxPooling2Layer : public ProtoLayer{
    public:
    int nf, nb;
    
    MaxPooling2Layer();
    void setAll();
    void forward();
    void backward();
};



class MaxPooling2x2Layer : public ProtoLayer{
    public:
    int nf, nb;
    
    MaxPooling2x2Layer();
    void setAll();
    void forward();
    void backward();
};




class FCLayer : public ProtoLayer{
    public:
    
    FCLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    void adjust(float alpha, float batchSize);

};

class FCDOLayer : public ProtoLayer{
    public:
    dataTensor* dropOutIdx;
    float* h_dropOutIdx;
    FCDOLayer();
    void setAll();
    void forward();
    void backward();
    dataTensor* get_dedw();
    
    void adjust(float alpha, float batchSize);

};





class FCBiasLayer : public ProtoLayer{    
    public:
    
    FCBiasLayer();
    void setAll();
    void forward();
    void forward(dataTensor* dropOutIdx);
    void backward();
    void backward(dataTensor* dropOutIdx);
    dataTensor* get_dedw();
    dataTensor* get_dedw(dataTensor* dropOutIdx);
    void adjust(float alpha, float batchSize);
};

class FCBRLayer : public ProtoLayer{// FC + FCBias + ReLU
    public:
    FCDOLayer* FC;
    FCBiasLayer* bias;
    ReLULayer* relu;
    dataTensor* FC2Bias;
    dataTensor* Bias2ReLU;
    dataTensor* Bias2FC;
    dataTensor* ReLU2Bias;
    dataTensor* FCWeight;
    dataTensor* BiasWeight;
    
    dataTensor* allWeight;

    //public:
    dataTensor* dropOutIdx;
    float* h_dropOutIdx;
    FCBRLayer();
    void setAll();
    void forward();
    void backward();
    
    void adjust(float alpha, float batchSize);
    
    dataTensor* getWeight();
    void loadWeight(int dataLength, float* in_weight);
};

class BindNtoOneLayer : public ProtoLayer{
// Do not set input and dedx.
    public:
    vector<dataTensor*> x_bundle;
    vector<dataTensor*> dedx_bundle;
    int inputN;
    int* accLength;
    
    BindNtoOneLayer();
    void setAll();// Call this function only after adding all input and dedx.
    void add_input(dataTensor* in_x);
    void add_dedx(dataTensor* dedx);
    void forward();
    void backward();
};

class StructBindLayer : public ProtoLayer{
// Do not set input and dedx.
    public:
    vector<dataTensor*> x_bundle;
    vector<dataTensor*> dedx_bundle;
    int inputN;
    int* accLength;
    
    StructBindLayer();
    void setAll();// Call this function only after adding all input and dedx.
    void add_input(dataTensor* in_x);
    void add_dedx(dataTensor* dedx);
    void forward();
    void backward();
};

class SumNtoOneLayer : public ProtoLayer{
// Do not set input and dedx. Use add_input and add_dedx instead.
    public:
    vector<dataTensor*> x_bundle;
    vector<dataTensor*> dedx_bundle;
    float** h_xHeads;
    float** d_xHeads;
    float** h_dedxHeads;
    
    int inputN;
    int x_bundleDimN;
    int* x_bundleDim;
    int featureLength;
    int bundleTotalLength;
    int bundleTotalFeatureN;
    
    SumNtoOneLayer();
    void setAll();// Call this function only after adding all input and dedx.
    void add_input(dataTensor* in_x);
    void add_dedx(dataTensor* dedx);
    void forward();
    void backward();
};

class SeparateOnetoNLayer : public ProtoLayer{
// Do not set output and dedy. Use add_dedy instead.
    public:
    vector<dataTensor*> dedy_bundle;
    int outputN;
    float** h_dedyHeads;
    float** d_dedyHeads;
    
    SeparateOnetoNLayer();
    void setAll();// Call this function only after adding all dedy.
    void add_dedy(dataTensor* dedy);
    void forward();
    void backward();
};

class NormalizeLayer : public ProtoLayer{
    public:
    int maxInd;
    float maxVal;
    int positive;
    float scale;
    
    NormalizeLayer();
    NormalizeLayer(float in_scale);
    void setAll();
    void forward();
    void backward();
};

class LogSoftmaxLayer : public ProtoLayer{
// Do not set output and dedy.
    public:
    float loss;
    float* h_in_x;
    float* h_dedx;
    int classNumber;
    double sum;
    double* expComp;
    
    
    LogSoftmaxLayer();
    
    void setAll();
    void forward();
    float forward(int rightClass);
    void backward();
    void backward(int rightClass);
    void showDist();
};

class diffSquareLayer : public ProtoLayer{
// Do not set output and dedy.
    public:
    float loss;
    float* h_in_x;
    float* h_dedx;
    int classNumber;
    double sum;
    double* expComp;
    
    
    diffSquareLayer();
    
    void setAll();
    void forward();
    float forward(int rightClass);
    void backward();
    void backward(int rightClass);
    void showDist();
};

class naiveLossLayer : public ProtoLayer{
// Do not set output and dedy.
    public:
    float loss;
    float* h_in_x;
    float* h_dedx;
    int classNumber;
    double sum;
    double* expComp;
    int minInd;
    double* subsOutput;
    
    
    naiveLossLayer();
    
    void setAll();
    void forward();
    float forward(int rightClass);
    void backward();
    void backward(int rightClass);
    void showDist();
};

class EucliDistLayer : public ProtoLayer{// Euclidean distance layer
// Do not set output and dedy.
    public:
    dataTensor* loss;
    dataTensor* tempDistance;
    dataTensor* square;
    dataTensor* tag;
    
    
    EucliDistLayer();
    
    void setAll();
    void forward();
    float forward(float* correctAns);
    void backward();
};



class inception3DPack{
    private:
    float fraction;
    
    dataTensor* oneConv1Alldedw;
    dataTensor* oneConv2Alldedw;
    dataTensor* oneConv3Alldedw;
    dataTensor* oneConv4Alldedw;
    dataTensor* threeConvAlldedw;
    dataTensor* fiveConvAlldedw;
    
    public:
    int* dimension;
    int dimensionN;
    
    int oneConv1FMN; // The number of feature maps of oneConv1;
    int oneConv2FMN;
    int oneConv3FMN;
    int oneConv4FMN;
    int threeConvFMN;
    int fiveConvFMN;
    
    Conv3DLayer* oneConv1;
    Conv3DLayer* oneConv2;
    Conv3DLayer* oneConv3;
    Conv3DLayer* oneConv4;
    Conv3DLayer* threeConv;
    Conv3DLayer* fiveConv;
    MaxPooling3DLayer* threeMaxPooling;
    SeparateOnetoNLayer* inputLayer;
    StructBindLayer* outputLayer;
    ReLULayer* afterOneConv1;
    ReLULayer* afterOneConv2;
    ReLULayer* afterOneConv3;
    ReLULayer* afterOneConv4;
    ReLULayer* afterThreeConv;
    ReLULayer* afterFiveConv;
    
    
    inception3DPack(int batchSize, int in_oneConv1FMN, int in_oneConv2FMN, int in_oneConv3FMN, int in_oneConv4FMN, int in_threeConvFMN, int in_fiveConvFMN); // the same with inputPackSize in FC_FCBias_ReLUPack.
    void setInput(dataTensor* x);
    void setOutput(dataTensor* y);
    void setdedx(dataTensor* dedx);
    void setdedy(dataTensor* dedy);
    void setAll();// setAll() only after setInput() and setOutput().
    
    void forward();
    void backward();
    void sumUp_dedw();
    void adjust(float alpha);
};

#endif
