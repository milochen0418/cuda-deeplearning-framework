#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include "kernel3.h"
#include <limits>
#include "myMath.h"
#include "dataTensor.h"

using namespace std;
    
    void dataTensor::setDataLength(){
        DL = 1;
        for(int i=0; i<dimensionN; i++){
            DL *= dimension[i];
        }
    }
    int dataTensor::dataLength(){
        return DL;
    }
    void dataTensor::generalInit(){
        cudaError_t errmsg;
        errmsg = cudaMalloc((void **) &d_data, dataLength()*sizeof(float));
        if(errmsg == cudaErrorMemoryAllocation){
            cout << "dataTensor::generalInit() cudaMalloc Error!\n";
            ALLBREAK = 1;
        }
        h_data = NULL;
        errmsg = cudaMalloc((void **) &d_dimension, dimensionN*sizeof(int));
        if(errmsg == cudaErrorMemoryAllocation){
            cout << "dataTensor::generalInit() cudaMalloc Error!\n";
            ALLBREAK = 1;
        }
        cudaMemcpy(d_dimension, dimension, dimensionN*sizeof(int), cudaMemcpyHostToDevice);
    }
    
    void dataTensor::randAssign(){// This function assign random values to d_data.
        if(enabled == true){
            if (h_data != NULL){
                delete[] h_data;
            }
            h_data = new float[DL];
            for(int i=0; i<dataLength(); i++){
                h_data[i] = (float)rand()/(float)RAND_MAX - 0.5;
            }
            cudaMemcpy(d_data, h_data, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_data;
            h_data = NULL;
        }else{
            cout << "randAssign() data only after initialized. Call constructor with parameters to initialize." << endl;
        }
    }
    void dataTensor::zeroAssign(){// This function assign zeros to d_data.
        if(enabled == true){
            if (h_data != NULL){
                delete[] h_data;
            }
            h_data = new float[DL];
            for(int i=0; i<dataLength(); i++){
                h_data[i] = 0.0;
            }
            cudaMemcpy(d_data, h_data, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_data;
            h_data = NULL;
        }else{
            cout << "zeroAssign() data only after initialized. Call constructor with parameters to initialize." << endl;
        }
    }
    
    void dataTensor::oneAssign(){// This function assign zeros to d_data.
        if(enabled == true){
            if (h_data != NULL){
                delete[] h_data;
            }
            h_data = new float[DL];
            for(int i=0; i<dataLength(); i++){
                h_data[i] = 1.0;
            }
            cudaMemcpy(d_data, h_data, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
            delete[] h_data;
            h_data = NULL;
        }else{
            cout << "zeroAssign() data only after initialized. Call constructor with parameters to initialize." << endl;
        }
    }
    
    void dataTensor::resetData(float* data_pointer){
        if(enabled == true){
            cudaMemcpy(d_data, data_pointer, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
        }else{
            cout << "resetData() only after initialized. Call constructor with parameters to initialize." << endl;
        }
    }
    
    dataTensor::dataTensor(){
        dimension = NULL;
        d_dimension = NULL;
        enabled = false;
    }
    dataTensor::dataTensor(int dimN, int* dim, float* data_pointer){
        
        dimensionN = dimN;
        dimension = dim;
        setDataLength();
        
        generalInit();
        
        cudaMemcpy(d_data, data_pointer, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
        enabled = true;
    }
    dataTensor::dataTensor(int dimN, int* dim){
        
        
        dimensionN = dimN;
        dimension = dim;
        
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
    }
    dataTensor::dataTensor(int dim_1, float* data_pointer){
        
        
        dimensionN = 1;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        setDataLength();
        generalInit();
        cudaMemcpy(d_data, data_pointer, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
        enabled = true;
    }
    dataTensor::dataTensor(int dim_1, int dim_2, float* data_pointer){
        
        dimensionN = 2;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        setDataLength();
        generalInit();
        cudaMemcpy(d_data, data_pointer, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
        enabled = true;
    }
    dataTensor::dataTensor(int dim_1, int dim_2, int dim_3, float* data_pointer){
        
        dimensionN = 3;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        dimension[2] = dim_3;
        setDataLength();
        generalInit();
        cudaMemcpy(d_data, data_pointer, dataLength()*sizeof(float), cudaMemcpyHostToDevice);
        enabled = true;
    }
    dataTensor::dataTensor(int dim_1, int dim_2, int dim_3, int dim_4, float* data_pointer){
        
        dimensionN = 4;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        dimension[2] = dim_3;
        dimension[3] = dim_4;
        setDataLength();
        generalInit();
        cudaMemcpy(d_data, data_pointer, DL*sizeof(float), cudaMemcpyHostToDevice);
        enabled = true;
    }
    dataTensor::dataTensor(int dim_1, int dim_2, int dim_3, int dim_4, int dim_5, float* data_pointer){
        
        dimensionN = 5;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        dimension[2] = dim_3;
        dimension[3] = dim_4;
        dimension[4] = dim_5;
        setDataLength();
        generalInit();
        cudaMemcpy(d_data, data_pointer, DL*sizeof(float), cudaMemcpyHostToDevice);
        enabled = true;
    }
    dataTensor::dataTensor(int dim_1){
        
        dimensionN = 1;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
    }
    dataTensor::dataTensor(int dim_1, int dim_2){
        
        dimensionN = 2;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
    }
    dataTensor::dataTensor(int dim_1, int dim_2, int dim_3){
        
        dimensionN = 3;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        dimension[2] = dim_3;
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
    }
    dataTensor::dataTensor(int dim_1, int dim_2, int dim_3, int dim_4){
        
        dimensionN = 4;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        dimension[2] = dim_3;
        dimension[3] = dim_4;
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
    }
    dataTensor::dataTensor(int dim_1, int dim_2, int dim_3, int dim_4, int dim_5){
        
        dimensionN = 5;
        dimension = new int[dimensionN];
        dimension[0] = dim_1;
        dimension[1] = dim_2;
        dimension[2] = dim_3;
        dimension[3] = dim_4;
        dimension[4] = dim_5;
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
    }
    
    void dataTensor::showDim(){
        if(enabled == true){
            cout << "Dimension info in host memory:\n";
            cout << "tensor dim: " << dimensionN << " ";
            for(int kk=0; kk<dimensionN; kk++){
                cout << dimension[kk] << " ";
            }
            cout << endl;
            
            int* tempDim = new int[dimensionN];
            cudaMemcpy(tempDim, d_dimension, dimensionN*sizeof(int), cudaMemcpyDeviceToHost);
            cout << "Dimension info in device memory:\n";
            cout << "tensor dim: " << dimensionN << " ";
            for(int kk=0; kk<dimensionN; kk++){
                cout << tempDim[kk] << " ";
            }
            cout << endl;
            
            delete[] tempDim;
            tempDim = NULL;
            
        }else{
            cout << "dataTensor::showDim() Not enabled yet.\n";
        }
    }
    
    dataTensor::~dataTensor(){
        cout << "dataTensor::~dataTensor()\n";
        if(enabled == true){
            if(h_data != NULL){
                delete[] h_data;
                h_data = NULL;
            }
            if(dimension != NULL){
                delete[] dimension;
                dimension = NULL;
            }
            cudaFree(d_data);
            cudaFree(d_dimension);
        }
    }
    
    void dataTensor::setTensor(int dimN, int* dim){
        if(enabled == true){
            cudaFree(d_data);
            cudaFree(d_dimension);
        }
        dimensionN = dimN;
        if(dimension != NULL){
            delete[] dimension;
        }
        dimension = new int[dimN];
        for(int i=0; i<dimN; i++){
            dimension[i] = dim[i];
        }
        setDataLength();
        generalInit();
        enabled = true;
        randAssign();
        
    }
    
    void dataTensor::setDim(int dimN, int* dim){
        dimensionN = dimN;
        if(dimension != NULL){
            delete[] dimension;
        }
        if(d_dimension != NULL){
            cudaFree(d_dimension);
        }
        dimension = new int[dimN];
        for(int i=0; i<dimN; i++){
            dimension[i] = dim[i];
        }
        cudaError_t errmsg;
        errmsg = cudaMalloc((void **) &d_dimension, dimensionN*sizeof(int));
        if(errmsg == cudaErrorMemoryAllocation){
            cout << "dataTensor::setDim() cudaMalloc Error!\n";
            ALLBREAK = 1;
        }
        
        cudaMemcpy(d_dimension, dimension, dimensionN*sizeof(int), cudaMemcpyHostToDevice);
    }
    
    float* dataTensor::getData(){// This function returns the pointer to the data in host.
        if(enabled == true){
            float* tempdata = new float[DL];
            cudaMemcpy(tempdata, d_data, dataLength()*sizeof(float), cudaMemcpyDeviceToHost);
            return(tempdata);
        }else{
            cout << "getData() only after initialized. Call constructor with parameters to initialize." << endl;
            return NULL;
        }
    }
    void dataTensor::showData(){
        if(enabled == true){
        if (h_data != NULL){
            delete[] h_data;
        }
        h_data = new float[DL];
        cudaMemcpy(h_data, d_data, dataLength()*sizeof(float), cudaMemcpyDeviceToHost);
        if(dimensionN == 1){
            int i;
            for(i=0; i<dimension[0]; i++){
                cout << h_data[i] << endl;
            }
            cout << endl;
        }else if(dimensionN == 2){
            int i,j;
            for(i=0; i<dimension[0]; i++){
                for(j=0; j<dimension[1]; j++){
                    cout << h_data[i + j*dimension[0]] << " ";
                }
                cout << endl;
            }
        }else if(dimensionN == 3){
            int i,j,k;
            for(i=0; i<dimension[2]; i++){
                cout << "feature map " << i+1 << ":\n";
                for(j=0; j<dimension[0]; j++){
                    for(k=0; k<dimension[1]; k++){
                        cout << h_data[j + k*dimension[0] + i*dimension[0]*dimension[1]] << " ";
                    }
                    cout << endl;
                }
                cout << endl;
            }
        }else if(dimensionN == 4){
            int i,j,k,m;
            for(m=0; m<dimension[3]; m++){
                cout << "feature map " << m+1 << ":\n";
                for(i=0; i<dimension[2]; i++){
                    cout << "layer " << i+1 << endl;
                    for(j=0; j<dimension[0]; j++){
                        for(k=0; k<dimension[1]; k++){
                            cout << h_data[j + k*dimension[0] + i*dimension[0]*dimension[1] + m*dimension[0]*dimension[1]*dimension[2]] << " ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;
            }
        }else if(dimensionN == 5){
            int i,j,k,m,n;
            for(n=0; n<dimension[4]; n++){
            for(m=0; m<dimension[3]; m++){
                cout << "feature map " << m+1 << ":\n";
                for(i=0; i<dimension[2]; i++){
                    cout << "layer " << i+1 << endl;
                    for(j=0; j<dimension[0]; j++){
                        for(k=0; k<dimension[1]; k++){
                            cout << h_data[j + k*dimension[0] + i*dimension[0]*dimension[1] + m*dimension[0]*dimension[1]*dimension[2] + n*dimension[0]*dimension[1]*dimension[2]*dimension[3]] << " ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                }
                cout << endl;
            }
            }
        }
        }else{
            cout << "showData() only after initialized. Call constructor with parameters to initialize." << endl;
        }
    }
    
    bool dataTensor::getEnabled(){
        return enabled;
    }
    
    
bool dataTensor::checkDimEqual(int dimN, int* dim){
    bool ans = true;
    int lengthA = 1;
    int lengthB = DL;
    
    for(int i=0; i<dimN; i++){
        lengthA *= dim[i];
    }
    
    if(dimensionN == dimN){
        for(int i=0; i<dimN; i++){
            if(dim[i] != dimension[i]){
                ans = false;
                break;
            }
        }
    }else{
        ans = false;
    }
    if(lengthA != lengthB){
        cout << "dataTensor::checkDimEqual(): Something wrong! data length not consistent.\n";
        cout << "lengthA: " << lengthA << " lengthB: " << lengthB << endl;
    }
    return(ans);
}
    
void dataTensor::save(string fileName){
    ofstream ofile(fileName.c_str(), ios::binary); 
    float* temp;
    ofile.write((char*) &dimensionN, sizeof(int));
    ofile.write((char*) dimension, sizeof(int)*dimensionN);
    temp = this->getData();
    ofile.write((char*) temp, sizeof(float)*(this->dataLength()));
    ofile.close();
    delete[] temp;
    temp = NULL;
}

void dataTensor::load(string fileName){
    if(enabled){
        ifstream ifile(fileName.c_str(), ios::binary);
        int i;
        float* temp;
        int tempDimN;
        int* tempDim;
        int length = 1;
        ifile.read((char*) &tempDimN, sizeof(int));
        tempDim = new int[tempDimN];
        ifile.read((char*) tempDim, sizeof(int)*tempDimN);
        for(i=0; i<tempDimN; i++){
            length *= tempDim[i];
        }
        if(length == this->dataLength()){
            if(tempDimN != this->dimensionN){
                cout << "dataTensor::load() warning: dimensionN not match.\n";
            }else{
                for(i=0; i<tempDimN; i++){
                    if(tempDim[i] != this->dimension[i]){
                        cout << "dataTensor::load() warning: dimension not match.\n";
                        break;
                    }
                }
            }
            temp = new float[length];
            ifile.read((char*) temp, sizeof(float)*length);
            ifile.close();
            this->resetData(temp);
            delete[] temp;
            temp = NULL;
        }else{
            cout << "dataTensor::load() Data length not match!\n";
        }
    }else{
        cout << "dataTensor::load() only after initialized. Call constructor with parameters to initialize." << endl;
    }
}
    
void tensorSum(dataTensor* result, dataTensor* component, float c){
    if(result->dataLength() == component->dataLength()){
        int BN = ceil((float) result->dataLength()/512.0);
        //GPUVectorScalarAddUp           (float* d_in, float* d_out, int dataL, float alpha)
        
        GPUVectorScalarAddUp<<<BN, 512>>>(component->d_data, result->d_data, result->dataLength(), c);
        
    }else{
        cout << "tensorSum(): Dimensions not consistent.\n";
    }
}

MatStruct dataTensor::getMatStruct(){
    vector<int> outDim;
    for (int i = 0; i < dimensionN; i++){
        outDim.push_back(dimension[i]);
    }

    MatStruct returnValue(outDim);
    float* tempData = this->getData();
    returnValue.set(tempData);
    delete[] tempData;
    return returnValue;
}

//=========================================================================
