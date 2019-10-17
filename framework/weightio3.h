#ifndef WEIGHTIO3
#define WEIGHTIO3

#include "kernel3.h"
#include "netStructTool3.h"
#include <map>
#include <vector>
#include <string>

void loadWeight(netStruct& net, string fileName);

void saveWeight(netStruct& net, string fileName);



class netDescriptor{
    private:
    void clearDescBundle();
    void clearWeightBundle();
        
    public:
    ~netDescriptor();
    map<int, layerDescriptor*> descBundle;
//    map<int, ProtoLayer*> net_inOrder;
//    map<int, ProtoLayer*> net_inId;
    map<int, int> id2order;
    map<int, int> order2id;
    vector<int> inputLayerId; // It is not the list of INPUT id, it is the inputLayers id.
    int outputLayerId;
    int lossLayerId;
    int classNumber;

    map<int, float*> weightBundle;// id oriented.
    map<int, int> weightLength;
    map<int, float*>::iterator weightBundleIt;

    map<string, MatStruct> weightPack;
    // map<string, MatStruct> flowPack;
    int extractNetInfo(netStruct& inNet); 
    int extractNetWeight(netStruct& inNet);
    void extractNetWeightV1(netStruct& inNet);
    // int extractNetFlow(netStruct& inNet); 
    // int extractNet(netStruct& inNet); // extract info, weight, and weights by calling the 3 functions above.    

    void loadWeightV1(string fileName);
    void loadWeightV1(ifstream& ifile);
    void saveWeightV1(string fileName);
    void saveWeightV1(ofstream& ofile);
    void putWeightV1(netStruct& inNet);

    void setDropOutRatio(int id, float ratio);
    
    
    int saveWeightV2(string fileName);
    int loadWeightV2(string fileName);
    // int saveFlow(string fileName);

    void putWeight(netStruct& inNet); // Put weight into inNet.
};

//void saveWeight2(netStruct& net, string fileName);
#endif
