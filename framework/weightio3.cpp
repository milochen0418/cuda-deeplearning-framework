#include <iostream>
#include <fstream>
//#include "kernel.h"
//#include "netStructTool.h"
#include <map>
#include <vector>
#include <string>
#include "weightio3.h"

#include <memory.h>

using namespace std;

void loadWeight(netStruct& net, string fileName){
    ifstream ifile(fileName.c_str(), ios::binary); 
    int version;
    int weightNum;
    int* id;
    int* weightLength;
    map<int, float*> weightBundle;// id oriented.
    int i, j;
    
    ifile.read((char*) &version, sizeof(int));
    ifile.read((char*) &weightNum, sizeof(int));
    
    id = new int[weightNum];
    weightLength = new int[weightNum];
    
    ifile.read((char*) id, sizeof(int)*weightNum);
    ifile.read((char*) weightLength, sizeof(int)*weightNum);
    
    
    
    for(i=0; i<weightNum; i++){
        weightBundle[id[i]] = new float[weightLength[i]];
        ifile.read((char*) (weightBundle[id[i]]), sizeof(float)*weightLength[i]);
    }
      
    ifile.close();
    
    map<int, float*>::iterator it;
    int count = 0;
    for(it=weightBundle.begin(); it != weightBundle.end(); it++){
        net.net_inId[it->first]->loadWeight(weightLength[count], it->second);
        delete [] it->second;
        it->second = NULL;
        count++;
    }
}

void saveWeight(netStruct& net, string fileName){
    
    map<int, ProtoLayer*>::iterator it;
    map<int, int> weightLength;
    int weightNum = 0;
    int version = 1;
    int id, length, i;
    map<int, float*> weightBundle;
    map<int, float*>::iterator bundleIt;
    
    
    for(it=net.net_inId.begin(); it != net.net_inId.end(); it++){
        if(it->second->layerType %7 == 6){
            dataTensor* tempWeight = it->second->getWeight();
            weightBundle[it->first] = tempWeight->getData();
            weightLength[it->first] = tempWeight->dataLength();
        }
    }
    weightNum = weightBundle.size();
    ofstream ofile(fileName.c_str(), ios::binary); 
    ofile.write((char*) &version, sizeof(int));
    ofile.write((char*) &weightNum, sizeof(int));
    for(bundleIt = weightBundle.begin(); bundleIt != weightBundle.end(); bundleIt++){
        id = bundleIt->first;
        ofile.write((char*) &id, sizeof(int));
    }
    
    for(bundleIt = weightBundle.begin(); bundleIt != weightBundle.end(); bundleIt++){
        length = weightLength[bundleIt->first];
        ofile.write((char*) &length, sizeof(int));
    }
    
    for(bundleIt = weightBundle.begin(); bundleIt != weightBundle.end(); bundleIt++){
        ofile.write((char*) (weightBundle[bundleIt->first]), sizeof(float)*weightLength[bundleIt->first]);
        delete[] weightBundle[bundleIt->first];
        //for(i=0; i<weightLength[bundleIt->first]; i++){
        //    ofile.write((char*) &(weightBundle[bundleIt->first][i]), sizeof(float));
        //}
    }
    
    
    ofile.close();
}


//============================================================================================

netDescriptor::~netDescriptor(){
   clearDescBundle();
}

void netDescriptor::clearDescBundle(){
    map<int, layerDescriptor*>::iterator iter;
    for(iter = descBundle.begin(); iter != descBundle.end(); iter++){
        if(iter->second != NULL){
            delete iter->second;
            iter->second = NULL;
        }
    }
    descBundle.clear();
}

void netDescriptor::setDropOutRatio(int id, float ratio){
    map<int, layerDescriptor*>::iterator iter = descBundle.find(id);
    if(iter != descBundle.end()){
        iter->second->dropOutRatio = ratio;
    }else{
        cout << "netDescriptor::setDropOutRatio(): id not found in this->descBundle." << endl;
    }
}




void netDescriptor::saveWeightV1(string fileName){
    ofstream ofile(fileName.c_str(), ios::binary); 
    this->saveWeightV1(ofile);
    ofile.close();
}

void netDescriptor::saveWeightV1(ofstream& ofile){
    
    int weightNum = 0;
    int version = 1;
    int id, length, i;    
    
    
    weightNum = weightBundle.size();
    
    ofile.write((char*) &version, sizeof(int));
    ofile.write((char*) &weightNum, sizeof(int));
    for(weightBundleIt = weightBundle.begin(); weightBundleIt != weightBundle.end(); weightBundleIt++){
        id = weightBundleIt->first;
        ofile.write((char*) &id, sizeof(int));
    }
    
    for(weightBundleIt = weightBundle.begin(); weightBundleIt != weightBundle.end(); weightBundleIt++){
        length = this->weightLength[weightBundleIt->first];
        ofile.write((char*) &length, sizeof(int));
    }
    
    for(weightBundleIt = weightBundle.begin(); weightBundleIt != weightBundle.end(); weightBundleIt++){
        ofile.write((char*) (weightBundle[weightBundleIt->first]), sizeof(float)*(this->weightLength[weightBundleIt->first]));
    }
}

void netDescriptor::loadWeightV1(string fileName){
    ifstream ifile(fileName.c_str(), ios::binary);
    loadWeightV1(ifile);
    ifile.close();
}

void netDescriptor::loadWeightV1(ifstream& ifile){
    int version;
    int weightNum;
    int* id;
    int* localWeightLength;
    int i, j;

    clearWeightBundle();
    weightLength.clear();
    
    ifile.read((char*) &version, sizeof(int));
    if(version != 1){
        cout << "netDescriptor::loadWeightV1(): version incorrect." << endl;
    }
    ifile.read((char*) &weightNum, sizeof(int));
    
    id = new int[weightNum];
    localWeightLength = new int[weightNum];
    
    ifile.read((char*) id, sizeof(int)*weightNum);
    ifile.read((char*) localWeightLength, sizeof(int)*weightNum);
    
    
    
    for(i=0; i<weightNum; i++){
        this->weightBundle[id[i]] = new float[localWeightLength[i]];
        this->weightLength[id[i]] = localWeightLength[i];
        ifile.read((char*) (this->weightBundle[id[i]]), sizeof(float)*localWeightLength[i]);
    }
    
    delete [] id;
    delete [] localWeightLength;
}

void netDescriptor::putWeightV1(netStruct& inNet){
    for(weightBundleIt=this->weightBundle.begin(); weightBundleIt != this->weightBundle.end(); weightBundleIt++){
        inNet.net_inId[weightBundleIt->first]->loadWeight(this->weightLength[weightBundleIt->first], weightBundleIt->second);
    }
}


int netDescriptor::saveWeightV2(string fileName){
    ofstream ofile(fileName.c_str(), ios::binary); 
    int layerNumber = descBundle.size();
    map<int, layerDescriptor*>::iterator iter;
    int id = 0;
    int version = 2;
    int i, tempInt, layerType; // i is used for loop index; tempInt is used to temporaly store many informations of layers.
    string strId; // id in a string type.
    char buffer[20];

    ofile.write((char*) &version, sizeof(int)); // Write out version of format.
    ofile.write((char*) &layerNumber, sizeof(int)); // Write out the number of layers.

    // Write out id of each layer.
    for(iter = descBundle.begin(); iter != descBundle.end(); iter++){
        id = iter->second->id;
        ofile.write((char*) &id, sizeof(int));
    }

    // Write out data (information and weights) of each layer.
    for(iter = descBundle.begin(); iter != descBundle.end(); iter++){
        // The part of writing out information.
        id = iter->second->id;
        ofile.write((char*) &id, sizeof(int));

        layerType = iter->second->layerType;
        ofile.write((char*) &layerType, sizeof(int));

        if(layerType != INPUT){
            tempInt = iter->second->sourceNum;
            ofile.write((char*) &tempInt, sizeof(int));
            ofile.write((char*) iter->second->sourceId, sizeof(int)*tempInt);
        }
        
        if((layerType != LOGSOFTMAX) && (layerType != EUCLIDIST)){
            tempInt = iter->second->destNum;
            ofile.write((char*) &tempInt, sizeof(int));
            ofile.write((char*) iter->second->destId, sizeof(int)*tempInt);
        }
        
        if(layerType != INPUT){
            tempInt = iter->second->inDimNum;
            ofile.write((char*) &tempInt, sizeof(int));
            ofile.write((char*) iter->second->inDim, sizeof(int)*tempInt);
        }

        if((layerType != LOGSOFTMAX) && (layerType != EUCLIDIST)){
            tempInt = iter->second->outDimNum;
            ofile.write((char*) &tempInt, sizeof(int));
            ofile.write((char*) iter->second->outDim, sizeof(int)*tempInt);
        }

        tempInt = (iter->second->isInput == true) ? 1 : 0;
        ofile.write((char*) &tempInt, sizeof(int));

        if((layerType%7 == 6) || (layerType == ZEROPADDING1D) || (layerType == ZEROPADDING2D) || (layerType == ZEROPADDING3D)){
            tempInt = iter->second->weightDimNum;
            ofile.write((char*) &tempInt, sizeof(int));
            ofile.write((char*) iter->second->weightDim, sizeof(int)*tempInt);
        }

        if((layerType == FCBR) || (layerType == FCDO)){
            ofile.write((char*) &(iter->second->dropOutRatio), sizeof(float));
        }

        // The part of writing out weights.
        if(layerType % 7 == 6){
            if((layerType == FCBR) || (layerType == CBR1D) || (layerType == CBR2D) || (layerType == CBR3D)){
                sprintf(buffer, "%d.1", id); // non bias layer weight
                strId = std::string(buffer);
                weightPack[strId].save(ofile);

                sprintf(buffer, "%d.2", id); // bias layer weight
                strId = std::string(buffer);
                weightPack[strId].save(ofile);
                
            }else{
                sprintf(buffer, "%d", id); // non bias layer weight
                strId = std::string(buffer);
                weightPack[strId].save(ofile);
            }
        }

    }

    ofile.close();
    return(1); // 1 means correct.
}

int netDescriptor::loadWeightV2(string fileName){
    clearDescBundle();
    id2order.clear();
    order2id.clear();
    inputLayerId.clear();
    weightPack.clear();

    int layerNumber = 0;
    map<int, layerDescriptor*>::iterator iter;
    int id = 0;
    int version = 2;
    int i, tempInt, layerType; // i is used for loop index; tempInt is used to temporaly store many informations of layers.
    string strId; // id in a string type.
    char buffer[20];
    vector<int> idVector;

    

    ifstream ifile(fileName.c_str(), ios::binary); 

    ifile.read((char*) &version, sizeof(int));

    if(version != 2){
        ifile.close();
        cout << "netDescriptor::loadWeightV2(): file version incorrect." << endl;
        return(2);
    }

    ifile.read((char*) &layerNumber, sizeof(int));

    for(i=0; i<layerNumber; i++){
        ifile.read((char*) &id, sizeof(int));
        idVector.push_back(id);
    }

    for(i=0; i<layerNumber; i++){
        layerDescriptor* tempLayer = new layerDescriptor;
        ifile.read((char*) &id, sizeof(int));
        tempLayer->id = id;

        ifile.read((char*) &layerType, sizeof(int));
        tempLayer->layerType = layerType;

        if(layerType != INPUT){
            ifile.read((char*) &tempInt, sizeof(int));
            tempLayer->sourceNum = tempInt;
            tempLayer->sourceId = new int[tempInt];

            ifile.read((char*) tempLayer->sourceId, sizeof(int)*tempInt);
        }

        if((layerType != LOGSOFTMAX) && (layerType != EUCLIDIST)){
            ifile.read((char*) &tempInt, sizeof(int));
            tempLayer->destNum = tempInt;
            tempLayer->destId = new int[tempInt];

            ifile.read((char*) tempLayer->destId, sizeof(int)*tempInt);
        }

        if(layerType != INPUT){
            ifile.read((char*) &tempInt, sizeof(int));
            tempLayer->inDimNum = tempInt;
            tempLayer->inDim = new int[tempInt];

            ifile.read((char*) tempLayer->inDim, sizeof(int)*tempInt);
        }
        
        if((layerType != LOGSOFTMAX) && (layerType != EUCLIDIST)){
            ifile.read((char*) &tempInt, sizeof(int));
            tempLayer->outDimNum = tempInt;
            tempLayer->outDim = new int[tempInt];

            ifile.read((char*) tempLayer->outDim, sizeof(int)*tempInt);
        }

        ifile.read((char*) &tempInt, sizeof(int));
        tempLayer->isInput = (tempInt == 1);

        if((layerType%7 == 6) || (layerType == ZEROPADDING1D) || (layerType == ZEROPADDING2D) || (layerType == ZEROPADDING3D)){
            ifile.read((char*) &tempInt, sizeof(int));
            tempLayer->weightDimNum = tempInt;
            tempLayer->weightDim = new int[tempInt];

            ifile.read((char*) tempLayer->weightDim, sizeof(int)*tempInt);
        }

        if((layerType == FCBR) || (layerType == FCDO)){
            ifile.read((char*) &(tempLayer->dropOutRatio), sizeof(float));
        }

        if(layerType % 7 == 6){
            if((layerType == FCBR) || (layerType == CBR1D) || (layerType == CBR2D) || (layerType == CBR3D)){
                sprintf(buffer, "%d.1", id); // non bias layer weight
                strId = std::string(buffer);
                weightPack[strId] = MatStruct();
                weightPack[strId].load(ifile);

                sprintf(buffer, "%d.2", id); // bias layer weight
                strId = std::string(buffer);
                weightPack[strId] = MatStruct();
                weightPack[strId].load(ifile);
            }else{
                sprintf(buffer, "%d", id); // non bias layer weight
                strId = std::string(buffer);
                weightPack[strId] = MatStruct();
                weightPack[strId].load(ifile);
            }
        }
        descBundle[id] = tempLayer;
    }

    ifile.close();
    return(1); // 1 means correct.
}

int netDescriptor::extractNetInfo(netStruct& inNet){
    clearDescBundle();
    id2order.clear();
    order2id.clear();
    inputLayerId.clear();
    map<int, layerDescriptor*>::iterator iter;

    id2order = inNet.id2order;
    order2id = inNet.order2id;
    inputLayerId = inNet.inputLayerId;
    outputLayerId = inNet.outputLayerId;
    classNumber = inNet.classNumber;

    for(iter = inNet.descBundle.begin(); iter != inNet.descBundle.end(); iter++){
        descBundle[iter->first] = new layerDescriptor; // In fact, iter->first is the id of this layer.
        *(descBundle[iter->first]) = *(iter->second);
    }
    return(1);
}

void netDescriptor::clearWeightBundle(){
    for(weightBundleIt=weightBundle.begin(); weightBundleIt != weightBundle.end(); weightBundleIt++){
        if(weightBundleIt->second != NULL){
            delete [] weightBundleIt->second;
        }
    }
    weightBundle.clear();
}

void netDescriptor::extractNetWeightV1(netStruct& inNet){
    dataTensor* tempWeight = NULL;
    map<int, ProtoLayer*>::iterator it;

    clearWeightBundle();

    for(it=inNet.net_inId.begin(); it != inNet.net_inId.end(); it++){
        if(it->second->layerType %7 == 6){
            tempWeight = it->second->getWeight();
            this->weightBundle[it->first] = tempWeight->getData();
            this->weightLength[it->first] = tempWeight->dataLength();
        }
    }
}

int netDescriptor::extractNetWeight(netStruct& inNet){


    weightPack.clear();
    map<int, ProtoLayer*>::iterator iter;
    dataTensor* tempTensor = NULL;
    float* tempWeight = NULL;
    int layerType = 0;
    int id = 0;
    int length = 1;
    int dimNum = 0;
    int i;
    vector<int> vectorDim;
    char buffer[50];
    MatStruct tempMat;
    string strId;


    for(iter = inNet.net_inId.begin(); iter != inNet.net_inId.end(); iter++){
        layerType = iter->second->layerType;
        if(layerType %7 == 6){
            id = iter->second->id;
            tempTensor = iter->second->getWeight();
            tempWeight = tempTensor->getData();

            if((layerType == CBR1D) || (layerType == CBR2D) || (layerType == CBR3D)){
                sprintf(buffer, "%d.1", id); // non bias layer weight
                strId = std::string(buffer);

                dimNum = descBundle[id]->weightDimNum;
                vectorDim.clear();
                length = 1;
                for(i=0; i<dimNum; i++){
                    vectorDim.push_back(descBundle[id]->weightDim[i]);
                    length *= descBundle[id]->weightDim[i];
                }

                vectorDim.push_back(descBundle[id]->inDim[dimNum]); // descBundle[id]->inDim[dimNum] is the number of input feature maps.
                vectorDim.push_back(descBundle[id]->outDim[dimNum]); // descBundle[id]->outDim[dimNum] is the number of output feature maps.

                weightPack[strId] = MatStruct(vectorDim);
                weightPack[strId].set(tempWeight);



                sprintf(buffer, "%d.2", id); // bias layer weight
                strId = std::string(buffer);

                weightPack[strId] = MatStruct(vectorDim.back());
                weightPack[strId].set(tempWeight + length);
                delete[] tempWeight;


                tempWeight = NULL;
            }else if(layerType == FCBR){


                sprintf(buffer, "%d.1", id); // non bias layer weight
                strId = std::string(buffer);

                dimNum = 2;
                vectorDim.clear();
                
                vectorDim.push_back(descBundle[id]->inDim[0]);
                vectorDim.push_back(descBundle[id]->outDim[0]);
                length = vectorDim[0]*vectorDim[1];

                weightPack[strId] = MatStruct(vectorDim);
                weightPack[strId].set(tempWeight);


                sprintf(buffer, "%d.2", id); // bias layer weight
                strId = std::string(buffer);

                weightPack[strId] = MatStruct(vectorDim[1]);
                weightPack[strId].set(tempWeight + length);
                delete[] tempWeight;


                tempWeight = NULL;
            }else if(layerType == FCDO){


                sprintf(buffer, "%d", id); // non bias layer weight
                strId = std::string(buffer);

                dimNum = 2;
                vectorDim.clear();
                
                vectorDim.push_back(descBundle[id]->inDim[0]);
                vectorDim.push_back(descBundle[id]->outDim[0]);
                length = vectorDim[0]*vectorDim[1];

                weightPack[strId] = MatStruct(vectorDim);
                weightPack[strId].set(tempWeight);

                delete[] tempWeight;


                tempWeight = NULL;
            }else{
                sprintf(buffer, "%d", id); // non bias layer weight
                strId = std::string(buffer);

                dimNum = descBundle[id]->weightDimNum;
                vectorDim.clear();
                length = 1;
                for(i=0; i<dimNum; i++){
                    vectorDim.push_back(descBundle[id]->weightDim[i]);
                    length *= descBundle[id]->weightDim[i];
                }

                weightPack[strId] = MatStruct(vectorDim);
                weightPack[strId].set(tempWeight);

                delete[] tempWeight;
                tempWeight = NULL;
            }
        }
    }
    return(1);
}

void netDescriptor::putWeight(netStruct& inNet){
    map<int, ProtoLayer*>::iterator iter;
    int id = 0;
    int layerType = 0;
    int length1 = 0;
    int length2 = 0;
    int length3 = 0;
    string strId1;
    string strId2;
    char buffer[50];
    float* data1 = NULL;
    float* data2 = NULL;
    float* data3 = NULL;

    for(iter = inNet.net_inId.begin(); iter != inNet.net_inId.end(); iter++){
        id = iter->first;
        layerType = iter->second->layerType;

        if(layerType %7 == 6){
            if((layerType == FCBR) || (layerType == CBR1D) || (layerType == CBR2D) || (layerType == CBR3D)){
                sprintf(buffer, "%d.1", id); // non bias layer weight
                strId1 = std::string(buffer);
                sprintf(buffer, "%d.2", id); // bias layer weight
                strId2 = std::string(buffer);

                length1 = this->weightPack[strId1].getLength();
                length2 = this->weightPack[strId2].getLength();
                length3 = length1 + length2;
                data1 = this->weightPack[strId1].getDataPt();
                data2 = this->weightPack[strId2].getDataPt();
                data3 = new float[length3];

                memcpy(data3, data1, sizeof(float)*length1);
                memcpy(data3 + length1, data2, sizeof(float)*length2);

                iter->second->loadWeight(length3, data3);

                delete [] data3; // No need to delete data1 and data2, because getDataPt() returns the pointer to the original data.
                data1 = NULL;
                data2 = NULL;
                data3 = NULL;
            }else{
                sprintf(buffer, "%d", id); // non bias layer weight
                strId1 = std::string(buffer);

                length1 = this->weightPack[strId1].getLength();
                data1 = this->weightPack[strId1].getDataPt();
                iter->second->loadWeight(length1, data1);

                data1 = NULL;
            }
        }
    }
}
