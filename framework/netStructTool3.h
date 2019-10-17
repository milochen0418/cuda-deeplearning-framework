#ifndef NETSTRUCTTOOL3
#define NETSTRUCTTOOL3

#include "kernel3.h"
#include <map>
#include <vector>
#include <string>
#include "matrixtool3.h"

class chart{
    public:
    int i;
    vector<int*> table;
    void add(int x, int y);
    int get(int x, int y);
    
    private:
    int search(int x, int y);
};

class layerDescriptor{
    public:
    int layerType;
    int id;
    int order;
    int sourceNum; // INPUT does not have this.
    int* sourceId; // INPUT does not have this.
    int destNum; // LOGSOFTMAX and EUCLIDIST do not have this.
    int* destId; // LOGSOFTMAX and EUCLIDIST do not have this.
    int inDimNum; // INPUT does not have this.
    int* inDim; // INPUT does not have this.
    int outDimNum; // LOGSOFTMAX and EUCLIDIST do not have this.
    int* outDim; // LOGSOFTMAX and EUCLIDIST do not have this.
    bool isInput;
    int weightDimNum; // Only layerType == 6 (mod 7) have this; padding uses this for padding information.
    int* weightDim; // Only layerType == 6 (mod 7) have this; padding uses this for padding information.
    string weightFile;
    float dropOutRatio; // Only FCDO and FCBR have this.
    
    bool checkLayerType;
    bool checkId;
    bool checkOrder;
    bool checkSource;
    bool checkDest;
    bool checkInputDim;
    bool checkOutputDim;
    bool checkIsInput;
    bool checkWeight;
    bool checkWeightFile;
    
    layerDescriptor();
    ~layerDescriptor();

    layerDescriptor& operator = (const layerDescriptor&);
};

class netStruct{
    public:
    map<int, layerDescriptor*> descBundle;
    map<int, ProtoLayer*> net_inOrder;
    map<int, ProtoLayer*> net_inId;
    map<int, int> id2order;
    map<int, int> order2id;
    vector<int> inputLayerId; // It is not the list of INPUT id, it is the inputLayers id.
    int outputLayerId;
    int lossLayerId;
    int classNumber;
};

netStruct constructNet(string fileName);

class inputStruct{
    public:
    map<int, float*> inputBundle;// inputBundle[layerId] == data.
    int rightClass;
    int resultClass;
    ~inputStruct();
};

class inputStruct2{
    public:
    map<int, MatStruct*> inputBundle;// inputBundle[layerId] == data.
    int rightClass;
    int resultClass;
    MatStruct* tagMatrix;
    ~inputStruct2();
};

class batchClass{
    private:
    int branchCount;
    vector<int> allLayerId;
    

    public:
    vector<inputStruct2*> inputStructBundle;
    vector<vector<float> > score;
    void addLayerId(int layerId);// Run this function multiple times to add multiple layer id.
    void addData(MatStruct& data);// Run this function only after add all layer id. Add data in the order the same with add layer id.
    void setDataTag(int);// Run this function only after add one data completely.
    void setDataTagMatrix(MatStruct&);
    void clearInputs();
    void clearLayerIds();
    int getBatchSize();
    int getResultTag(int);
    int getRightTag(int);
    int getCorrectCount();
    batchClass();
    //~batchClass();
};



float playNet(netStruct&, vector<inputStruct*>&, float alpha);

float playNet(netStruct&, vector<inputStruct2*>&, float alpha);

float playNet(netStruct&, vector<inputStruct*>&);

float playNet(netStruct&, vector<inputStruct2*>&);

float playNet(netStruct&, vector<inputStruct2*>&, bool computeGrad, bool displayScore);

float playNet(netStruct&, batchClass&, bool computeGrad, bool displayScore);

float playNetForTest(netStruct&, batchClass&, bool computeGrad, bool displayScore);

void adjustNet(netStruct& net, float alpha, int batchSize);

void testNet(netStruct&, vector<inputStruct*>&, int tagNumber);

int testNet(netStruct&, inputStruct*, int tagNumber); // This function returns a tag.

int testNet(netStruct&, inputStruct2*, int tagNumber); // This function returns a tag.

void showNetConnection(netStruct&);




#endif
