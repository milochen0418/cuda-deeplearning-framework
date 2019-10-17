#include <iostream> // for cout
//#include "kernel.h"
#include "netStructTool3.h"
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <math.h>
#include <sstream>


#include <stdio.h>
#include <stdlib.h>

using namespace std;

inputStruct::~inputStruct(){
    map<int, float*>::iterator it;
    for(it = inputBundle.begin(); it != inputBundle.end(); it++){
        if(it->second != NULL)
        delete[] it->second;
    }
}

inputStruct2::~inputStruct2(){
    map<int, MatStruct*>::iterator it;
    for(it = inputBundle.begin(); it != inputBundle.end(); it++){
        if(it->second != NULL){
            //cout << "inputStruct2::~inputStruct2() delete inputBundle " << it->first << endl;
            delete it->second;
        }
        
    }
}

void chart::add(int x, int y){
    if(search(x, y) > -1){
        cout << "Index already exists.\n";
    }else{
        int* temp = new int[2];
        temp[0] = x;
        temp[1] = y;
        table.push_back(temp);
    }
}

    int chart::get(int x, int y){
        int ans = search(x, y);
        if(ans == -1){
            cout << "No such index.\n";
        }
        return(ans);
    }
    
    int chart::search(int x, int y){
        int ans = -1;
        for(i=0; i<table.size(); i++){
            if((table[i][0] == x) && (table[i][1] == y)){
                ans = i;
                break;
            }
        }
        return(ans);
    }

    layerDescriptor::layerDescriptor(){
        sourceId = NULL;
        destId = NULL;
        inDim = NULL;
        outDim = NULL;
        weightDim = NULL;

        layerType = 0;
        id = 0;
        order = 0;
        sourceNum = 0;
        destNum = 0;
        inDimNum = 0;
        outDimNum = 0;
        weightDimNum = 0;

        checkLayerType = false;
        checkId = false;
        checkOrder = false;
        checkSource = false;
        checkDest = false;
        checkInputDim = false;
        checkOutputDim = false;
        checkWeight = false;
        checkWeightFile = false;
        isInput = false;
        dropOutRatio = 0.0;
    }

    layerDescriptor::~layerDescriptor(){
        if(sourceId != NULL) 
            delete [] sourceId;
        if(destId != NULL)
            delete [] destId;
        if(inDim != NULL)
            delete [] inDim;
        if(outDim != NULL)
            delete [] outDim;
        if(weightDim != NULL)
            delete [] weightDim;
    }

layerDescriptor& layerDescriptor::operator = (const layerDescriptor& inDesc){
    int i = 0;
    if(this->sourceId != NULL) 
        delete [] this->sourceId;
    if(this->destId != NULL)
        delete [] this->destId;
    if(this->inDim != NULL)
        delete [] this->inDim;
    if(this->outDim != NULL)
        delete [] this->outDim;
    if(this->weightDim != NULL)
        delete [] this->weightDim;
    
    this->layerType = inDesc.layerType;
    this->id = inDesc.id;
    this->order = inDesc.order;

    if(inDesc.sourceNum > 0){
        this->sourceNum = inDesc.sourceNum;
        this->sourceId = new int[inDesc.sourceNum];
        for(i=0; i<inDesc.sourceNum; i++){
            this->sourceId[i] = inDesc.sourceId[i];
        }
    }else{
        this->sourceId = NULL;
    }

    if(inDesc.destNum > 0){
        this->destNum = inDesc.destNum;
        this->destId = new int[inDesc.destNum];
        for(i=0; i<inDesc.destNum; i++){
            this->destId[i] = inDesc.destId[i];
        }
    }else{
        this->destId = NULL;
    }

    if(inDesc.inDimNum > 0){
        this->inDimNum = inDesc.inDimNum;
        this->inDim = new int[inDesc.inDimNum];
        for(i=0; i<inDesc.inDimNum; i++){
            this->inDim[i] = inDesc.inDim[i];
        }
    }else{
        this->inDim = NULL;
    }

    if(inDesc.outDimNum > 0){
        this->outDimNum = inDesc.outDimNum;
        this->outDim = new int[inDesc.outDimNum];
        for(i=0; i<inDesc.outDimNum; i++){
            this->outDim[i] = inDesc.outDim[i];
        }
    }else{
        this->outDim = NULL;
    }

    if(inDesc.weightDimNum > 0){
        this->weightDimNum = inDesc.weightDimNum;
        this->weightDim = new int[inDesc.weightDimNum];
        for(i=0; i<inDesc.weightDimNum; i++){
            this->weightDim[i] = inDesc.weightDim[i];
        }
    }else{
        this->weightDim = NULL;
    }

    this->isInput = inDesc.isInput;
    this->dropOutRatio = inDesc.dropOutRatio;
    return(*this);
}


void arrangeOrder(netStruct& myNet){
    map<int, layerDescriptor*>& descBundle = myNet.descBundle;
    map<int, bool> orderOK;
    map<int, layerDescriptor*>::iterator i;
    map<int, bool>::iterator k;
    layerDescriptor* tempDesc;
    bool allTrue = false;
    int j, count;
    int orderCount = 0;
    
    int safeCount = 0;
    
    for(i=descBundle.begin(); i != descBundle.end(); i++){
        if(i->second->layerType == INPUT){
            orderOK[i->first] = true;
        }else{
            orderOK[i->first] = false;
        }
    }
    
    while(!allTrue){
        for(i = descBundle.begin(); i != descBundle.end(); i++){
            if(!orderOK[i->first]){
                tempDesc = i->second;
                count = 0;
                for(j=0; j<tempDesc->sourceNum; j++){
                    if(orderOK[tempDesc->sourceId[j]]){
                        count++;
                    }
                }
                if(count == tempDesc->sourceNum){
                    orderOK[i->first] = true;
                    tempDesc->order = orderCount;
                    tempDesc->checkOrder = true;
                    myNet.order2id[orderCount] = i->first;
                    myNet.id2order[i->first] = orderCount;
                    orderCount++;
                }
            }
        }
        
        allTrue = true;
        for(k=orderOK.begin(); k != orderOK.end(); k++){
            if(k->second == false){
                allTrue = false;
                break;
            }
        }
        safeCount++;
        if(safeCount > 300){
            cout << "safeCount > 300: " << safeCount << endl;
        }
        
    }
    
}

vector<int> getNumbers(char* buffer){
    vector<int> ans;
    stringstream ss;
    ss.str(buffer);
    int tempint;
    while(ss >> tempint){
        ans.push_back(tempint);
    }
    
    return(ans);
}

void loadDescriptor(map<int, layerDescriptor*>& descBundle, vector<int>& idVector, string fileName){
    //string fileName = "CC001.txt";
    ifstream netFile;
    layerDescriptor* tempDesc;
    string tempStr;
    int tempInt;
    char* buffer = new char[200];
    vector<int> numbers;
    
    netFile.open(fileName.c_str());
        while(netFile >> tempStr){
            if(tempStr == "begin"){
                tempDesc = new layerDescriptor;
                netFile >> tempStr;
                while(tempStr != "end"){
                    if(tempStr == "type"){
                        netFile >> tempStr;
                        if(tempStr == "CONV2D"){
                            tempDesc->layerType = CONV2D;
                        }else if(tempStr == "CONV3D"){
                            tempDesc->layerType = CONV3D;
                        }else if(tempStr == "CBR1D"){
                            tempDesc->layerType = CBR1D;
                        }else if(tempStr == "CBR2D"){
                            tempDesc->layerType = CBR2D;
                        }else if(tempStr == "CBR3D"){
                            tempDesc->layerType = CBR3D;
                        }else if(tempStr == "ZEROPADDING1D"){
                            tempDesc->layerType = ZEROPADDING1D;
                        }else if(tempStr == "ZEROPADDING2D"){
                            tempDesc->layerType = ZEROPADDING2D;
                        }else if(tempStr == "ZEROPADDING3D"){
                            tempDesc->layerType = ZEROPADDING3D;
                        }else if(tempStr == "CONV2DBIAS"){
                            tempDesc->layerType = CONV2DBIAS;
                        }else if(tempStr == "CONV3DBIAS"){
                            tempDesc->layerType = CONV3DBIAS;
                        }else if(tempStr == "FC"){
                            tempDesc->layerType = FC;
                        }else if(tempStr == "FCDO"){
                            tempDesc->layerType = FCDO;
                        }else if(tempStr == "FCBR"){
                            tempDesc->layerType = FCBR;
                        }else if(tempStr == "FCBIAS"){
                            tempDesc->layerType = FCBIAS;
                        }else if(tempStr == "MAXPOOLING2"){
                            tempDesc->layerType = MAXPOOLING2;
                        }else if(tempStr == "MAXPOOLING2X2"){
                            tempDesc->layerType = MAXPOOLING2X2;
                        }else if(tempStr == "MAXPOOLING2X2X2"){
                            tempDesc->layerType = MAXPOOLING2X2X2;
                        }else if(tempStr == "MAXPOOLING2D"){
                            tempDesc->layerType = MAXPOOLING2D;
                        }else if(tempStr == "MAXPOOLING3D"){
                            tempDesc->layerType = MAXPOOLING3D;
                        }else if(tempStr == "NORMALIZE"){
                            tempDesc->layerType = NORMALIZE;
                        }else if(tempStr == "RELU"){
                            tempDesc->layerType = RELU;
                        }else if(tempStr == "LOGSOFTMAX"){
                            tempDesc->layerType = LOGSOFTMAX;
                        }else if(tempStr == "NAIVELOSS"){
                            tempDesc->layerType = NAIVELOSS;
                        }else if(tempStr == "DIFFSQUARE"){
                            tempDesc->layerType = DIFFSQUARE;
                        }else if(tempStr == "BINDNTOONE"){
                            tempDesc->layerType = BINDNTOONE;
                        }else if(tempStr == "SUMNTOONE"){
                            tempDesc->layerType = SUMNTOONE;
                        }else if(tempStr == "SEPARATEONETON"){
                            tempDesc->layerType = SEPARATEONETON;
                        }else if(tempStr == "PROTO"){
                            tempDesc->layerType = PROTO;
                        }else if(tempStr == "INPUT"){
                            tempDesc->layerType = INPUT;
                        }else if(tempStr == "EUCLIDIST"){
                            tempDesc->layerType = EUCLIDIST;
                        }
                        
                        tempDesc->checkLayerType = true;
                    }else if(tempStr == "id"){
                        netFile >> tempDesc->id;
                        tempDesc->checkId = true;
                    }else if(tempStr == "dropOutRatio"){
                        netFile >> tempDesc->dropOutRatio;
                        //tempDesc->checkId = true;
                    }else if(tempStr == "sourceId"){
                        netFile.getline(buffer, 199);
                        numbers = getNumbers(buffer);
                        tempDesc->sourceNum = numbers.size();
                        tempDesc->sourceId = new int[tempDesc->sourceNum];
                        for(int i=0; i<tempDesc->sourceNum; i++){
                            tempDesc->sourceId[i] = numbers[i];
                        }
                        tempDesc->checkSource = true;
                    }else if(tempStr == "destId"){
                        netFile.getline(buffer, 199);
                        numbers = getNumbers(buffer);
                        tempDesc->destNum = numbers.size();
                        tempDesc->destId = new int[tempDesc->destNum];
                        for(int i=0; i<tempDesc->destNum; i++){
                            tempDesc->destId[i] = numbers[i];
                        }
                        tempDesc->checkDest = true;
                    }else if(tempStr == "inDim"){
                        netFile.getline(buffer, 199);
                        numbers = getNumbers(buffer);
                        tempDesc->inDimNum = numbers.size();
                        if(tempDesc->inDimNum > 0){
                            tempDesc->inDim = new int[tempDesc->inDimNum];
                            for(int i=0; i<tempDesc->inDimNum; i++){
                                tempDesc->inDim[i] = numbers[i];
                            }
                        }
                        tempDesc->checkInputDim = true;
                    }else if(tempStr == "outDim"){
                        netFile.getline(buffer, 199);
                        numbers = getNumbers(buffer);
                        tempDesc->outDimNum = numbers.size();
                        //cout << "loadDescriptor(): desc id " << tempDesc->id << endl;
                        //cout << "outDim: ";
                        if(tempDesc->outDimNum > 0){
                            tempDesc->outDim = new int[tempDesc->outDimNum];
                            for(int i=0; i<tempDesc->outDimNum; i++){
                                tempDesc->outDim[i] = numbers[i];
                                //cout << tempDesc->outDim[i] << " ";
                            }
                        }
                        //cout << endl;
                        tempDesc->checkOutputDim = true;
                    }else if(tempStr == "weightDim"){
                        netFile.getline(buffer, 199);
                        numbers = getNumbers(buffer);
                        tempDesc->weightDimNum = numbers.size();
                        tempDesc->weightDim = new int[tempDesc->weightDimNum];
                        for(int i=0; i<tempDesc->weightDimNum; i++){
                            tempDesc->weightDim[i] = numbers[i];
                        }
                        tempDesc->checkWeight = true;
                    }else if(tempStr == "padding"){
                        netFile.getline(buffer, 199);
                        numbers = getNumbers(buffer);
                        tempDesc->weightDimNum = numbers.size();
                        tempDesc->weightDim = new int[tempDesc->weightDimNum];
                        for(int i=0; i<tempDesc->weightDimNum; i++){
                            tempDesc->weightDim[i] = numbers[i];
                        }
                        tempDesc->checkWeight = true;
                    }else if(tempStr == "weightFile"){
                        netFile >> tempDesc->weightFile;
                        tempDesc->checkWeightFile = true;
                    }
                    netFile >> tempStr;
                }
                idVector.push_back(tempDesc->id);
                descBundle[tempDesc->id] = tempDesc;
            }
        }
        netFile.close();
        delete[] buffer;
}

netStruct constructNet(string fileName){
    netStruct myNet;
    map<int, layerDescriptor*>& descBundle = myNet.descBundle;
    layerDescriptor* tempDesc;

    int i, j, k, fromId, toId, tempId, tempOrder;
    vector<int> idVector;
    map<int, dataTensor*> forwardBundle;
    map<int, dataTensor*> backwardBundle;
    chart tensorIndex;
    map<int, ProtoLayer*>& net = myNet.net_inOrder;
    
    int tensorId = 1;
    
    // Load file to descBundle and idVector.
    loadDescriptor(descBundle, idVector, fileName);
    
        arrangeOrder(myNet);

        
        // Construct data tensor.
        // First, construct the bundles.
        for(i=0; i<idVector.size(); i++){
            fromId = idVector[i];
            if(descBundle[fromId]->checkDest){
                for(k=0; k<descBundle[fromId]->destNum; k++){
                    toId = descBundle[fromId]->destId[k];
                    //cout << "from: " << fromId << " to: " << toId << endl;
                    tensorIndex.add(fromId, toId);
                    forwardBundle[tensorIndex.get(fromId, toId)] = new dataTensor();
                    forwardBundle[tensorIndex.get(fromId, toId)]->id = tensorId;
                    
                    tensorId++;
                    backwardBundle[tensorIndex.get(fromId, toId)] = new dataTensor();
                    backwardBundle[tensorIndex.get(fromId, toId)]->id = tensorId;
                    //cout << "tensor id: " << tensorId << endl;
                    tensorId++;
                }
            }
        }

        //cout << "connection number: " << tensorIndex.table.size() << endl;
        
        
        // Second, initialize the tensor bundles.
        for(i=0; i<tensorIndex.table.size(); i++){
        
        
            fromId = tensorIndex.table[i][0];
            toId = tensorIndex.table[i][1];
            
            if(descBundle[fromId]->layerType == INPUT){
                myNet.inputLayerId.push_back(toId);
            }
            
            if(descBundle[toId]->layerType %7 == 1){// Layers without output.
                //myNet.outputLayerId.push_back(fromId);
                //myNet.lossLayerId.push_back(toId);
                myNet.outputLayerId = fromId;
                myNet.lossLayerId = toId;
                myNet.classNumber = descBundle[toId]->outDimNum;
            }
            
            
            // Do forward bundle.
            if(descBundle[fromId]->layerType == SEPARATEONETON){
                int tempFromId = fromId;
                int tempToId = toId;
                while(descBundle[tempFromId]->layerType == SEPARATEONETON){
                    tempToId = tempFromId;
                    tempFromId = descBundle[tempFromId]->sourceId[0];
                }
                delete forwardBundle[tensorIndex.get(fromId, toId)];
                forwardBundle[tensorIndex.get(fromId, toId)] = forwardBundle[tensorIndex.get(tempFromId, tempToId)];
            }else{
                forwardBundle[tensorIndex.get(fromId, toId)]->setTensor(descBundle[fromId]->outDimNum, descBundle[fromId]->outDim);
            }
            
            
            // Do backward bundle.
            /*if(descBundle[fromId]->layerType == SEPARATEONETON){
                backwardBundle[tensorIndex.get(fromId, toId)]->setTensor(descBundle[toId]->inDimNum, descBundle[toId]->inDim);
            }else if(descBundle[fromId]->layerType != INPUT){
                backwardBundle[tensorIndex.get(fromId, toId)]->setTensor(descBundle[fromId]->outDimNum, descBundle[fromId]->outDim);
            }else{
                delete backwardBundle[tensorIndex.get(fromId, toId)];
            }*/
            
            if(descBundle[fromId]->layerType == SEPARATEONETON){
                backwardBundle[tensorIndex.get(fromId, toId)]->setTensor(descBundle[toId]->inDimNum, descBundle[toId]->inDim);
            }else{
                backwardBundle[tensorIndex.get(fromId, toId)]->setTensor(descBundle[fromId]->outDimNum, descBundle[fromId]->outDim);
            }
        }
        
        for(i=0; i<tensorIndex.table.size(); i++){
            fromId = tensorIndex.table[i][0];
            toId = tensorIndex.table[i][1];
            if((descBundle[toId]->layerType == CONV1DBIAS) || (descBundle[toId]->layerType == CONV2DBIAS) || (descBundle[toId]->layerType == CONV3DBIAS) || (descBundle[toId]->layerType == FCBIAS)){
                //delete backwardBundle[tensorIndex.get(fromId, toId)];
                backwardBundle[tensorIndex.get(fromId, toId)] = backwardBundle[tensorIndex.get(toId, descBundle[toId]->destId[0])];
            }
        }
        
        // Third, construct the layers.
    for(i=0; i<idVector.size(); i++){
    
        tempId = idVector[i];
        tempDesc = descBundle[tempId];
        
        /*cout << "constructNet(): tempDesc->inDimNum: " << tempDesc->inDimNum << endl;
        cout << "constructNet(): tempDesc->inDim: ";
        for(int j=0; j<tempDesc->inDimNum; j++){
            cout << tempDesc->inDim[j] << " ";
        }
        cout << endl;
        cout << "constructNet(): tempDesc->outDimNum: " << tempDesc->outDimNum << endl;
        cout << "constructNet(): tempDesc->inDim: ";
        for(int j=0; j<tempDesc->outDimNum; j++){
            cout << tempDesc->outDim[j] << " ";
        }
        cout << endl;*/
        
        tempOrder = tempDesc->order;
        
//        cout << "constructNet(): layerType: " << tempDesc->layerType << endl;
        if(tempDesc->layerType != INPUT){
            if(tempDesc->layerType == LOGSOFTMAX){
                net[tempOrder] = new LogSoftmaxLayer;
                fromId = tempDesc->sourceId[0];
                
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
            }else if(tempDesc->layerType == DIFFSQUARE){
                net[tempOrder] = new diffSquareLayer;
                fromId = tempDesc->sourceId[0];
                
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
            }else if(tempDesc->layerType == EUCLIDIST){
            cout << "constructNet(): here 8" << endl;
                net[tempOrder] = new EucliDistLayer;
                fromId = tempDesc->sourceId[0];
                
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
            }else if(tempDesc->layerType == NAIVELOSS){
                net[tempOrder] = new naiveLossLayer;
                fromId = tempDesc->sourceId[0];
                
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
            }else if(tempDesc->layerType == BINDNTOONE){
                net[tempOrder] = new BindNtoOneLayer;
                toId = tempDesc->destId[0];
                for(j=0; j<tempDesc->sourceNum; j++){
                    net[tempOrder]->add_input(forwardBundle[tensorIndex.get(tempDesc->sourceId[j], tempId)]);
                    net[tempOrder]->add_dedx(backwardBundle[tensorIndex.get(tempDesc->sourceId[j], tempId)]);
                }
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
            }else if(tempDesc->layerType == SUMNTOONE){
                net[tempOrder] = new SumNtoOneLayer;
                toId = tempDesc->destId[0];
                for(j=0; j<tempDesc->sourceNum; j++){
                    net[tempOrder]->add_input(forwardBundle[tensorIndex.get(tempDesc->sourceId[j], tempId)]);
                    net[tempOrder]->add_dedx(backwardBundle[tensorIndex.get(tempDesc->sourceId[j], tempId)]);
                }
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
            }else if(tempDesc->layerType == SEPARATEONETON){
                net[tempOrder] = new SeparateOnetoNLayer;
                fromId = tempDesc->sourceId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                for(j=0; j<tempDesc->destNum; j++){
                    net[tempOrder]->add_dedy(backwardBundle[tensorIndex.get(tempId, tempDesc->destId[j])]);
                }
            }else if(tempDesc->layerType == CONV1D){
                net[tempOrder] = new Conv1DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                if(tempDesc->checkWeightFile){
                    //load weight file
                }else{
                    int* weightDim = tempDesc->weightDim;
                    ((Conv1DLayer*) net[tempOrder])->setWeight(weightDim[0]);
                }
            }else if(tempDesc->layerType == CONV2D){
                net[tempOrder] = new Conv2DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                if(tempDesc->checkWeightFile){
                    //load weight file
                }else{
                    int* weightDim = tempDesc->weightDim;
                    ((Conv2DLayer*) net[tempOrder])->setWeight(weightDim[0], weightDim[1]);
                }
            }else if(tempDesc->layerType == CONV3D){
                net[tempOrder] = new Conv3DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                if(tempDesc->checkWeightFile){
                    //load weight file
                }else{
                    int* weightDim = tempDesc->weightDim;
                    ((Conv3DLayer*) net[tempOrder])->setWeight(weightDim[0], weightDim[1], weightDim[2]);
                }
            }else if(tempDesc->layerType == CBR2D){
                net[tempOrder] = new CBR2DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                if(tempDesc->checkWeightFile){
                    //load weight file
                }else{
                    
                    int* weightDim = tempDesc->weightDim;
                    ((CBR2DLayer*) net[tempOrder])->setWeight(weightDim[0], weightDim[1]);
                }
            }else if(tempDesc->layerType == CBR1D){
                net[tempOrder] = new CBR1DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                if(tempDesc->checkWeightFile){
                    //load weight file
                }else{
                    int* weightDim = tempDesc->weightDim;
                    ((CBR1DLayer*) net[tempOrder])->setWeight(weightDim[0]);
                }
            }else if(tempDesc->layerType == CBR3D){
                net[tempOrder] = new CBR3DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                if(tempDesc->checkWeightFile){
                    //load weight file
                }else{
                    int* weightDim = tempDesc->weightDim;
                    ((CBR3DLayer*) net[tempOrder])->setWeight(weightDim[0], weightDim[1], weightDim[2]);
                }
            }else if(tempDesc->layerType == ZEROPADDING1D){
                
                net[tempOrder] = new ZeroPad1DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                int* padding = tempDesc->weightDim;
                ((ZeroPad1DLayer*) net[tempOrder])->setPadding(padding[0], padding[1]);
            }else if(tempDesc->layerType == ZEROPADDING2D){
                net[tempOrder] = new ZeroPad2DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                int* padding = tempDesc->weightDim;
                ((ZeroPad2DLayer*) net[tempOrder])->setPadding(padding[0], padding[1], padding[2], padding[3]);
            }else if(tempDesc->layerType == ZEROPADDING3D){
                net[tempOrder] = new ZeroPad3DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                int* padding = tempDesc->weightDim;
                ((ZeroPad3DLayer*) net[tempOrder])->setPadding(padding[0], padding[1], padding[2], padding[3], padding[4], padding[5]);
            }else if(tempDesc->layerType == CONV2DBIAS){
                net[tempOrder] = new Conv2DBiasLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == CONV3DBIAS){
                net[tempOrder] = new Conv3DBiasLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == FC){
                net[tempOrder] = new FCLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
            }else if(tempDesc->layerType == FCDO){
                net[tempOrder] = new FCDOLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setDropOutRatio(tempDesc->dropOutRatio); 
            }else if(tempDesc->layerType == FCBR){
                net[tempOrder] = new FCBRLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setDropOutRatio(tempDesc->dropOutRatio); 
            }else if(tempDesc->layerType == FCBIAS){
                net[tempOrder] = new FCBiasLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
            }else if(tempDesc->layerType == MAXPOOLING2){
                net[tempOrder] = new MaxPooling2Layer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == MAXPOOLING2X2){
                net[tempOrder] = new MaxPooling2x2Layer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == MAXPOOLING2X2X2){
                net[tempOrder] = new MaxPooling2x2x2Layer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == NORMALIZE){
                net[tempOrder] = new NormalizeLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == RELU){
                net[tempOrder] = new ReLULayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
            }else if(tempDesc->layerType == MAXPOOLING2D){
                net[tempOrder] = new MaxPooling2DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                ((MaxPooling2DLayer*) net[tempOrder])->setFilterDim(tempDesc->weightDim[0], tempDesc->weightDim[1]);
            }else if(tempDesc->layerType == MAXPOOLING3D){
                net[tempOrder] = new MaxPooling3DLayer;
                fromId = tempDesc->sourceId[0];
                toId = tempDesc->destId[0];
                net[tempOrder]->setInput(forwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setOutput(forwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setdedx(backwardBundle[tensorIndex.get(fromId, tempId)]);
                net[tempOrder]->setdedy(backwardBundle[tensorIndex.get(tempId, toId)]);
                net[tempOrder]->setInputDim(tempDesc->inDimNum, tempDesc->inDim);
                net[tempOrder]->setOutputDim(tempDesc->outDimNum, tempDesc->outDim);
                ((MaxPooling3DLayer*) net[tempOrder])->setFilterDim(tempDesc->weightDim[0], tempDesc->weightDim[1], tempDesc->weightDim[2]);
            }
            net[tempOrder]->id = tempId;
            
            /*cout << "constructNet(): layer id: " << tempId << endl;
            if(net[tempOrder]->input != NULL){
                cout << "constructNet(): before setAll() input dimension\n";
                net[tempOrder]->input->showDim();
            }
            if(net[tempOrder]->output != NULL){
                cout << "constructNet(): before setAll() output dimension\n";
                net[tempOrder]->output->showDim();
            }*/
            
            net[tempOrder]->setAll();
            
            /*if(net[tempOrder]->input != NULL){
                cout << "constructNet(): after setAll() input dimension\n";
                net[tempOrder]->input->showDim();
            }
            if(net[tempOrder]->output != NULL){
                cout << "constructNet(): after setAll() output dimension\n";
                net[tempOrder]->output->showDim();
            }*/
            
            myNet.net_inId[tempId] = net[tempOrder];
        }else{
            descBundle[tempDesc->destId[0]]->isInput = true;
        }
        
    }
    return(myNet);
}

float playNet(netStruct& myNet, vector<inputStruct*>& inputStructBundle, float alpha){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    //fraction = 0;
    //cout << "inputStructBundle.size() " << inputStructBundle.size() << endl;
    //cout << "fraction " << fraction << endl;
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    
    
            
    float* temp;
    float* temp1;
    
    for(int j=0; j<inputStructBundle.size(); j++){
        
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            //cout << "myNet.inputLayerId[i]: " << myNet.inputLayerId[i] << endl;
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]);
        }
        
      


        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                //cout << "forwarding id " << it->second->id << endl;
                it->second->forward();
                
             
                
            }else{
                //cout << "forwarding id " << it->second->id << endl;
                loss += it->second->forward(inputStructBundle[j]->rightClass);
                
                temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                tempMax = temp_h_in_x[0];
                tempMaxIndex = 0;
                
                cout << "Score of class 0: " << tempMax << endl;
                
                for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                    cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                    if(tempMax < temp_h_in_x[k]){
                        tempMax = temp_h_in_x[k];
                        tempMaxIndex = k;
                    }
                }
                cout << endl;
                
                
                cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                cout << "Computed tag: " << tempMaxIndex << endl;
                cout << endl;
                if(tempMaxIndex == inputStructBundle[j]->rightClass){
                    count++;
                }
            }
        }
        
        
        for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
            if(myNet.descBundle[rit->second->id]->isInput == true){
                // Do nothing.
            }else if(rit->second->id != myNet.lossLayerId){
                rit->second->backward();
            }else{
                rit->second->backward(inputStructBundle[j]->rightClass);
            }
        }
        
        
        
        
        
    }
    //cout << "loss: " << loss << endl;
    
    
    
    for(it = myNet.net_inId.begin(); it != myNet.net_inId.end(); it++){
        if(it->second->layerType %7 == 6){
            it->second->adjust(alpha, (float) inputStructBundle.size());
        }
    }
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
    return(loss);
}

float playNet(netStruct& myNet, vector<inputStruct*>& inputStructBundle){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    //fraction = 0;
    //cout << "inputStructBundle.size() " << inputStructBundle.size() << endl;
    //cout << "fraction " << fraction << endl;
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    
    
            
    float* temp;
    float* temp1;
    
    for(int j=0; j<inputStructBundle.size(); j++){
        
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            //cout << "myNet.inputLayerId[i]: " << myNet.inputLayerId[i] << endl;
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]);
        }
        
        

        // Play forward
        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                it->second->forward();
                
            }else{
                //cout << "forwarding id " << it->second->id << endl;
                loss += it->second->forward(inputStructBundle[j]->rightClass);
                
                temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                tempMax = temp_h_in_x[0];
                tempMaxIndex = 0;
                
                cout << "Score of class 0: " << tempMax << endl;
                
                for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                    cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                    if(tempMax < temp_h_in_x[k]){
                        tempMax = temp_h_in_x[k];
                        tempMaxIndex = k;
                    }
                }
                cout << endl;
                
                
                cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                cout << "Computed tag: " << tempMaxIndex << endl;
                cout << endl;
                inputStructBundle[j]->resultClass = tempMaxIndex;
                if(tempMaxIndex == inputStructBundle[j]->rightClass){
                    count++;
                }
            }
        }
        
        // Play backward
        for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
            if(myNet.descBundle[rit->second->id]->isInput == true){
                // Do nothing.
            }else if(rit->second->id != myNet.lossLayerId){
                rit->second->backward();
            }else{
                rit->second->backward(inputStructBundle[j]->rightClass);
            }
        }
        
        
        
        
    }
    //cout << "loss: " << loss << endl;
    
    
    
    
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
    return(loss);
}



float playNet(netStruct& myNet, vector<inputStruct2*>& inputStructBundle, float alpha){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    
    
            
    float* temp;
    float* temp1;
    
    
    for(int j=0; j<inputStructBundle.size(); j++){
        
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            //cout << "myNet.inputLayerId[i]: " << myNet.inputLayerId[i] << endl;
            //inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->showDim();
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
        }
        
        
        

        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            
            if(it->second->id != myNet.lossLayerId){
                
                it->second->forward();
                
                
            }else{
                //cout << "playNet(): it->second->id " << it->second->id << endl;
                
                loss += it->second->forward(inputStructBundle[j]->rightClass);
                
                
                
                temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                tempMax = temp_h_in_x[0];
                tempMaxIndex = 0;
                
                
                //cout << "Score of class 0: " << tempMax << endl;
                
                for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                    //cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                    if(tempMax < temp_h_in_x[k]){
                        tempMax = temp_h_in_x[k];
                        tempMaxIndex = k;
                    }
                }
                //cout << endl;
                
                
                //cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                //cout << "Computed tag: " << tempMaxIndex << endl;
                //cout << endl;
                if(tempMaxIndex == inputStructBundle[j]->rightClass){
                    count++;
                }
            }
        }
        
        
        
        for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
            if(myNet.descBundle[rit->second->id]->isInput == true){
                // Do nothing.
            }else if(rit->second->id != myNet.lossLayerId){
                rit->second->backward();
            }else{
                rit->second->backward(inputStructBundle[j]->rightClass);
            }
        }
        
        
    }
    
    
    
    for(it = myNet.net_inId.begin(); it != myNet.net_inId.end(); it++){
        if(it->second->layerType %7 == 6){
            it->second->adjust(alpha, (float) inputStructBundle.size());
        }
    }
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
    return(loss);
}

float playNet(netStruct& myNet, vector<inputStruct2*>& inputStructBundle){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    float* temp;
    float* temp1;
    
    
    for(int j=0; j<inputStructBundle.size(); j++){
        
        
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            //cout << "myNet.inputLayerId[i]: " << myNet.inputLayerId[i] << endl;
            //inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->showDim();
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
        }
        
        

        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                it->second->forward();
                //it->second->output->showData();
            }else{
                //cout << "playNet(): it->second->id " << it->second->id << endl;
                loss += it->second->forward(inputStructBundle[j]->rightClass);
                
                
                
                temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                tempMax = temp_h_in_x[0];
                tempMaxIndex = 0;
                
                
                cout << "Score of class 0: " << tempMax << endl;
                
                for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                    cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                    if(tempMax < temp_h_in_x[k]){
                        tempMax = temp_h_in_x[k];
                        tempMaxIndex = k;
                    }
                }
                cout << endl;
                
                
                cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                cout << "Computed tag: " << tempMaxIndex << endl;
                cout << endl;
                inputStructBundle[j]->resultClass = tempMaxIndex;
                if(tempMaxIndex == inputStructBundle[j]->rightClass){
                    count++;
                }
            }
        }
        
        
        
        for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
            if(myNet.descBundle[rit->second->id]->isInput == true){
                // Do nothing.
            }else if(rit->second->id != myNet.lossLayerId){
                rit->second->backward();
            }else{
                rit->second->backward(inputStructBundle[j]->rightClass);
            }
        }
        
    }
    
    
    
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
    return(loss);
}

float playNet(netStruct& myNet, vector<inputStruct2*>& inputStructBundle, bool computeGrad, bool displayScore){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    float* temp;
    float* temp1;
    
    if (displayScore){
        for(int j=0; j<inputStructBundle.size(); j++){
            
            
            //Set input.
            for(int i=0; i<myNet.inputLayerId.size(); i++){
                myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
            }
            
            

            for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
                if(it->second->id != myNet.lossLayerId){
                    it->second->forward();
                }else{
                    loss += it->second->forward(inputStructBundle[j]->rightClass);
                    
                    temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                    tempMax = temp_h_in_x[0];
                    tempMaxIndex = 0;
                    
                    cout << "Score of class 0: " << tempMax << endl;
                    
                    for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                        cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                        if(tempMax < temp_h_in_x[k]){
                            tempMax = temp_h_in_x[k];
                            tempMaxIndex = k;
                        }
                    }
                    cout << endl;
                    
                    
                    cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                    cout << "Computed tag: " << tempMaxIndex << endl;
                    cout << endl;
                    inputStructBundle[j]->resultClass = tempMaxIndex;
                    if(tempMaxIndex == inputStructBundle[j]->rightClass){
                        count++;
                    }
                }
            }
            
            if (computeGrad){
                for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
                    if(myNet.descBundle[rit->second->id]->isInput == true){
                        // Do nothing.
                        //cout << "input layer id: " << rit->second->id << endl;
                    }else if(rit->second->id != myNet.lossLayerId){
                        rit->second->backward();
                    }else{
                        rit->second->backward(inputStructBundle[j]->rightClass);
                    }
                }
            }
        }
    }else{
        for(int j=0; j<inputStructBundle.size(); j++){
            
            
            //Set input.
            for(int i=0; i<myNet.inputLayerId.size(); i++){
                myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
            }
            
            

            for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
                if(it->second->id != myNet.lossLayerId){
                    it->second->forward();
                }else{
                    loss += it->second->forward(inputStructBundle[j]->rightClass);
                    
                    temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                    tempMax = temp_h_in_x[0];
                    tempMaxIndex = 0;
                    
                    for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                        if(tempMax < temp_h_in_x[k]){
                            tempMax = temp_h_in_x[k];
                            tempMaxIndex = k;
                        }
                    }
                    cout << endl;
                    
                    
                    cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                    cout << "Computed tag: " << tempMaxIndex << endl;
                    cout << endl;
                    inputStructBundle[j]->resultClass = tempMaxIndex;
                    if(tempMaxIndex == inputStructBundle[j]->rightClass){
                        count++;
                    }
                }
            }
            
            if (computeGrad){
                for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
                    if(myNet.descBundle[rit->second->id]->isInput == true){
                        // Do nothing.
                        //cout << "input layer id: " << rit->second->id << endl;
                    }else if(rit->second->id != myNet.lossLayerId){
                        rit->second->backward();
                    }else{
                        rit->second->backward(inputStructBundle[j]->rightClass);
                    }
                }
            }
        }
    }
    
    
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
    return(loss);
}

float playNet(netStruct& myNet, batchClass& theBatch, bool computeGrad, bool displayScore){
    vector<inputStruct2*>& inputStructBundle = theBatch.inputStructBundle;
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    float localLoss;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    float* temp;
    float* temp1;
    vector<float> tempV;
    theBatch.score.assign(inputStructBundle.size(), tempV);
    
    if (displayScore){
//    cout << "playNet(): inputStructBundle.size() " << inputStructBundle.size() << endl;
        for(int j=0; j<inputStructBundle.size(); j++){
//            cout << "playNet(): inputStructBundle.size() count " << j << endl;
            
            
            //Set input.
            for(int i=0; i<myNet.inputLayerId.size(); i++){
                myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
            }
            
            

            for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
                if(it->second->id != myNet.lossLayerId){
                    it->second->forward();
                }else if(it->second->layerType == LOGSOFTMAX){
                    loss += it->second->forward(inputStructBundle[j]->rightClass);
                    
                    temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                    tempMax = temp_h_in_x[0];
                    tempMaxIndex = 0;
                    
                    cout << "Score of class 0: " << tempMax << endl;
                    theBatch.score[j].push_back(tempMax);
                    
                    for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                        cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                        theBatch.score[j].push_back(temp_h_in_x[k]);
                        if(tempMax < temp_h_in_x[k]){
                            tempMax = temp_h_in_x[k];
                            tempMaxIndex = k;
                        }
                    }
                    cout << endl;
                    
                    
                    cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                    cout << "Computed tag: " << tempMaxIndex << endl;
                    cout << endl;
                    inputStructBundle[j]->resultClass = tempMaxIndex;
                    if(tempMaxIndex == inputStructBundle[j]->rightClass){
                        count++;
                    }
                }else if(it->second->layerType == EUCLIDIST){
                    localLoss = it->second->forward(inputStructBundle[j]->tagMatrix->getDataPt());
                    loss += localLoss;
                    cout << "Loss of sample " << j << ": " << localLoss << "." << endl;
                }
            }
            
//            cout << "playNet(): here 1\n";
            
            if (computeGrad){
            
//            cout << "playNet(): here 2\n";
                for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
                
                    if(myNet.descBundle[rit->second->id]->isInput == true){
                        // Do nothing.
                        //cout << "input layer id: " << rit->second->id << endl;
                    }else if(rit->second->id != myNet.lossLayerId){
                        //cout << "playNet(): general layer grad.\n";
                        rit->second->backward();
                    }else if(rit->second->layerType == LOGSOFTMAX){
                        //cout << "playNet(): LOGSOFTMAX layer grad.\n";
                        rit->second->backward(inputStructBundle[j]->rightClass);
                    }else if(rit->second->layerType == EUCLIDIST){
                        //cout << "playNet(): EUCLIDIST layer grad.\n";
                        rit->second->backward();
                    }
                }
            }
//            cout << "playNet(): here 3\n";
        }
    }else{
        for(int j=0; j<inputStructBundle.size(); j++){
//            cout << "playNet(): inputStructBundle.size() count " << j << endl;
            
            
            //Set input.
            for(int i=0; i<myNet.inputLayerId.size(); i++){
                myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
            }
            
            

            for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
                if(it->second->id != myNet.lossLayerId){
                    it->second->forward();
                }else if(it->second->layerType == LOGSOFTMAX){
                    loss += it->second->forward(inputStructBundle[j]->rightClass);
                    
                    temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                    tempMax = temp_h_in_x[0];
                    tempMaxIndex = 0;
                    
                    theBatch.score[j].push_back(tempMax);
                    
                    for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                        theBatch.score[j].push_back(temp_h_in_x[k]);
                        if(tempMax < temp_h_in_x[k]){
                            tempMax = temp_h_in_x[k];
                            tempMaxIndex = k;
                        }
                    }
                    
                    
                    
                    inputStructBundle[j]->resultClass = tempMaxIndex;
                    if(tempMaxIndex == inputStructBundle[j]->rightClass){
                        count++;
                    }
                }else if(it->second->layerType == EUCLIDIST){
                    localLoss = it->second->forward(inputStructBundle[j]->tagMatrix->getDataPt());
                    loss += localLoss;
                    cout << "Loss of sample " << j << ": " << localLoss << "." << endl;
                }
            }
            
//            cout << "playNet(): here 1\n";
            
            if (computeGrad){
            
//            cout << "playNet(): here 2\n";
                for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
                
                    if(myNet.descBundle[rit->second->id]->isInput == true){
                        // Do nothing.
                        //cout << "input layer id: " << rit->second->id << endl;
                    }else if(rit->second->id != myNet.lossLayerId){
                        //cout << "playNet(): general layer grad.\n";
                        rit->second->backward();
                    }else if(rit->second->layerType == LOGSOFTMAX){
                        //cout << "playNet(): LOGSOFTMAX layer grad.\n";
                        rit->second->backward(inputStructBundle[j]->rightClass);
                    }else if(rit->second->layerType == EUCLIDIST){
                        //cout << "playNet(): EUCLIDIST layer grad.\n";
                        rit->second->backward();
                    }
                }
            }
//            cout << "playNet(): here 3\n";
        }
    }
    
//    cout << "playNet(): here 4\n";
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
//    cout << "playNet(): here 5\n";
    return(loss);
}


float playNetForTest(netStruct& myNet, batchClass& theBatch, bool computeGrad, bool displayScore){
    vector<inputStruct2*>& inputStructBundle = theBatch.inputStructBundle;
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    float fraction = 1.0/inputStructBundle.size();
    map<int, dataTensor*>::iterator dtit;
    float loss = 0.0;
    int count = 0;
    int usless;
    float tempMax, tempMaxIndex;
    int k;
    float* temp_h_in_x;
    float* temp;
    float* temp1;
    vector<float> tempV;
    theBatch.score.assign(inputStructBundle.size(), tempV);
    
    if (displayScore){
        for(int j=0; j<inputStructBundle.size(); j++){
            
            
            //Set input.
            for(int i=0; i<myNet.inputLayerId.size(); i++){
                myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
            }
            
            

            for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
                if(it->second->id != myNet.lossLayerId){
                    it->second->forward();
                }else{
                    loss += it->second->forward(inputStructBundle[j]->rightClass);
                    
                    temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                    tempMax = temp_h_in_x[0];
                    tempMaxIndex = 0;
                    
                    cout << "Score of class 0: " << tempMax << endl;
                    theBatch.score[j].push_back(tempMax);
                    
                    for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                        cout << "Score of class " << k << ": " << temp_h_in_x[k] << endl;
                        theBatch.score[j].push_back(temp_h_in_x[k]);
                        if(tempMax < temp_h_in_x[k]){
                            tempMax = temp_h_in_x[k];
                            tempMaxIndex = k;
                        }
                    }
                    cout << endl;
                    
                    
                    cout << "Right tag: " << inputStructBundle[j]->rightClass << " ";
                    cout << "Computed tag: " << tempMaxIndex << endl;
                    cout << endl;
                    inputStructBundle[j]->resultClass = tempMaxIndex;
                    if(tempMaxIndex == inputStructBundle[j]->rightClass){
                        count++;
                    }
                }
            }
            
            if (computeGrad){
                for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
                    if(rit->second->id != myNet.lossLayerId){
                        rit->second->backward();
                    }else{
                        rit->second->backward(inputStructBundle[j]->rightClass);
                    }
                }
            }
        }
    }else{
        for(int j=0; j<inputStructBundle.size(); j++){
            
            
            //Set input.
            for(int i=0; i<myNet.inputLayerId.size(); i++){
                myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]->getDataPt());
            }
            
            

            for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
                if(it->second->id != myNet.lossLayerId){
                    it->second->forward();
                }else{
                    loss += it->second->forward(inputStructBundle[j]->rightClass);
                    
                    temp_h_in_x = ((LogSoftmaxLayer*) it->second)->h_in_x;
                    tempMax = temp_h_in_x[0];
                    tempMaxIndex = 0;
                    theBatch.score[j].push_back(tempMax);
                    
                    for(k=1; k< ((LogSoftmaxLayer*) it->second)->classNumber; k++){
                        theBatch.score[j].push_back(temp_h_in_x[k]);
                        if(tempMax < temp_h_in_x[k]){
                            tempMax = temp_h_in_x[k];
                            tempMaxIndex = k;
                        }
                    }
                    inputStructBundle[j]->resultClass = tempMaxIndex;
                    if(tempMaxIndex == inputStructBundle[j]->rightClass){
                        count++;
                    }
                }
            }
            
            if (computeGrad){
                for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
                    if(rit->second->id != myNet.lossLayerId){
                        rit->second->backward();
                    }else{
                        rit->second->backward(inputStructBundle[j]->rightClass);
                    }
                }
            }
        }
    }
    
    
    cout << "correct count: " << count << "/" << inputStructBundle.size() << " == " << (float) count/(float) inputStructBundle.size() << endl;
    return(loss);
}

void adjustNet(netStruct& myNet, float alpha, int batchSize){

    map<int, ProtoLayer*>::iterator it;
    for(it = myNet.net_inId.begin(); it != myNet.net_inId.end(); it++){
        if(it->second->layerType %7 == 6){
            it->second->adjust(alpha, (float) batchSize);
        }
    }
    
}

/*
void playNet(netStruct& myNet, inputStruct& myInputStruct, float alpha){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(myInputStruct.inputBundle[myNet.inputLayerId[i]]);
        }

        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                it->second->forward();
            }else{
                cout << "right class: " << myInputStruct.rightClass << endl;
                it->second->forward(myInputStruct.rightClass);
            }
        }
        
        
        
        
        for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
            if(myNet.descBundle[rit->second->id]->isInput == true){
                // Do nothing.
            }else if(rit->second->id != myNet.lossLayerId){
                rit->second->backward();
            }else{
                rit->second->backward(myInputStruct.rightClass);
            }
        }
        
        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->layerType %7 == 6){
                   
                    it->second->adjust(alpha);
            }
            
        }
        
    
}*/

int findMaxClass(float* data, int classNum){
    int ans = 0;
    for(int i=1; i<classNum; i++){
        if(data[i] > data[ans]){
            ans = i;
        }
    }
    delete data;
    return(ans);
}

void testNet(netStruct& myNet, vector<inputStruct*>& inputStructBundle, int tagNumber){// tagNumber is the number of classes.
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    
    map<int, dataTensor*>::iterator dtit;
    
    int count = 0;
    int usless;
    int resultClass = 0;
    
    
    
    for(int j=0; j<inputStructBundle.size(); j++){
        
        
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(inputStructBundle[j]->inputBundle[myNet.inputLayerId[i]]);
        }



        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                it->second->forward();
            }else{
                cout << "Right class: " << inputStructBundle[j]->rightClass << endl;
                resultClass = findMaxClass(it->second->input->getData(), tagNumber);
                cout << "Result class: " << resultClass << endl;
                if(resultClass == inputStructBundle[j]->rightClass){count++;}
                
                
            }
        }
        
        
    }
    cout << "Correct ratio: " << count/((float) inputStructBundle.size()) << endl;
    
}

int testNet(netStruct& myNet, inputStruct* myInputStruct, int tagNumber){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    
    map<int, dataTensor*>::iterator dtit;
    
    int usless;
    int resultClass = 0;
    
    
    
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(myInputStruct->inputBundle[myNet.inputLayerId[i]]);
        }


        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                it->second->forward();
            }else{
                resultClass = findMaxClass(it->second->input->getData(), tagNumber);
                //cout << "Result class: " << resultClass << endl;
            }
        }
        
        return(resultClass);
    
}

int testNet(netStruct& myNet, inputStruct2* myInputStruct, int tagNumber){
    map<int, ProtoLayer*>::iterator it;
    map<int, ProtoLayer*>::reverse_iterator rit;
    
    map<int, dataTensor*>::iterator dtit;
    
    int usless;
    int resultClass = 0;
    
      
    
    
        //Set input.
        for(int i=0; i<myNet.inputLayerId.size(); i++){
            myNet.net_inId[myNet.inputLayerId[i]]->input->resetData(myInputStruct->inputBundle[myNet.inputLayerId[i]]->getDataPt());
        }
        


        for(it = myNet.net_inOrder.begin(); it != myNet.net_inOrder.end(); it++){
            if(it->second->id != myNet.lossLayerId){
                it->second->forward();
            }else{
                resultClass = findMaxClass(it->second->input->getData(), tagNumber);
                //cout << "Result class: " << resultClass << endl;
            }
        }
        
        
        return(resultClass);
    
}

void showNetConnection(netStruct& net){
// This function shows the connection of the network.
    map<int, vector<inputStruct*>* > BIG;
    
    map<int, ProtoLayer*>::iterator it;
    
    cout << "The connections of the network.\n";
    cout << "input tensor id; layer id; output tensor id.\n";
    cout << "Forward flow." << endl;
    
    for(it=net.net_inId.begin(); it != net.net_inId.end(); it++){
        if(it->second->layerType %7 == 2){// no input
        }else if(it->second->layerType %7 == 1){// no output
        }else if(it->second->layerType %7 == 4){// multiple output
        }else if(it->second->layerType %7 == 3){// multiple input
        }else{
            
            cout << it->second->input->id << " " << it->first << " " << it->second->output->id << endl;
        }
    }
    
    cout << "Backward flow." << endl;
    for(it=net.net_inId.begin(); it != net.net_inId.end(); it++){
        if(it->second->layerType %7 == 1){
        }else if(it->second->layerType %7 == 2){
        }else if(it->second->layerType %7 == 4){// multiple output
        }else if(it->second->layerType %7 == 3){// multiple input
        }else{
            cout << it->second->dedx->id << " " << it->first << " " << it->second->dedy->id << endl;
        }
    }
    cout << endl;
    
    
    
}

batchClass::batchClass(){
    branchCount = 0;
}

void batchClass::addLayerId(int layerId){
    if(inputStructBundle.size() == 0){
        allLayerId.push_back(layerId);
    }else{
        cout << "batchClass::addLayerId() You have already add some data.\n";
    }
}

void batchClass::addData(MatStruct& branchData){
    if(allLayerId.size() > 0){
        inputStruct2* tempInputStruct;
        if(branchCount == 0){
            tempInputStruct = new inputStruct2;
            inputStructBundle.push_back(tempInputStruct);
        }
//        cout << "batchClass::addData(): branch number " << branchCount << endl;
        MatStruct* tempMat = MatCpy(&branchData);
        tempInputStruct = inputStructBundle.back();
        tempInputStruct->inputBundle[allLayerId[branchCount]] = tempMat;
        branchCount++;
        if(branchCount == allLayerId.size()){
            branchCount = 0;
        }
    }else{
        cout << "batchClass::addData() Please add layer id first.\n";
    }
}

void batchClass::setDataTag(int tag){
    if(inputStructBundle.size() > 0){
        inputStruct2* tempInputStruct = inputStructBundle.back();
        tempInputStruct->rightClass = tag;
    }else{
        cout << "batchClass::setDataTag() Please add data first.\n";
    }
}

void batchClass::setDataTagMatrix(MatStruct& inputTagMatrix){
    if(inputStructBundle.size() > 0){
        inputStruct2* tempInputStruct = inputStructBundle.back();
        tempInputStruct->tagMatrix = &inputTagMatrix;
    }else{
        cout << "batchClass::setDataTagMatrix() Please add data first.\n";
    }
}

void batchClass::clearInputs(){
    for(int i=0; i<inputStructBundle.size(); i++){
        //cout << "batchClass::clearInputs() delete inputStructBundle " << i << endl;
        delete inputStructBundle[i];
    }
    inputStructBundle.clear();
    score.clear();
}

int batchClass::getBatchSize(){
    return(inputStructBundle.size());
}

int batchClass::getResultTag(int inputIdx){
    int ans = -1;
    if(inputIdx < inputStructBundle.size()){
        ans = inputStructBundle[inputIdx]->resultClass;
    }else{
        cout << "batchClass::getResultTag() The input index does not exist.\n";
    }
    return(ans);
}

int batchClass::getRightTag(int inputIdx){
    int ans = -1;
    if(inputIdx < inputStructBundle.size()){
        ans = inputStructBundle[inputIdx]->rightClass;
    }else{
        cout << "batchClass::getRightTag() The input index does not exist.\n";
    }
    return(ans);
}

int batchClass::getCorrectCount(){
    int ans = 0;
    for(int i=0; i<getBatchSize(); i++){
        if(getResultTag(i) ==  getRightTag(i)){
            ans++;
        }
    }
    return(ans);
}


//===========================================================================================

