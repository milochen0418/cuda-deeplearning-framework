//#include "kernel.h"
//#include <map>
#include <vector>
#include "matrixtool3.h"
#include <iostream> // for cout
#include <fstream>
//#include <math.h> // for fmod
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

vector<int> v1(int dim1){
    vector<int> ans;
    ans.push_back(dim1);
    return(ans);
}

vector<int> v2(int dim1, int dim2){
    vector<int> ans;
    ans.push_back(dim1);
    ans.push_back(dim2);
    return(ans);
}

vector<int> v3(int dim1, int dim2, int dim3){
    vector<int> ans;
    ans.push_back(dim1);
    ans.push_back(dim2);
    ans.push_back(dim3);
    return(ans);
}

vector<int> v4(int dim1, int dim2, int dim3, int dim4){
    vector<int> ans;
    ans.push_back(dim1);
    ans.push_back(dim2);
    ans.push_back(dim3);
    ans.push_back(dim4);
    return(ans);
}

vector<int> v5(int dim1, int dim2, int dim3, int dim4, int dim5){
    vector<int> ans;
    ans.push_back(dim1);
    ans.push_back(dim2);
    ans.push_back(dim3);
    ans.push_back(dim4);
    ans.push_back(dim5);
    return(ans);
}

int MatStruct::MatCount = 0;

void MatStruct::setDim(vector<int> newDimension){
    if(data != NULL){
        delete[] data;
        data = NULL;
    }
    dimension = newDimension;
    length = 1;
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

/*vector<float> MatStruct::toVector(){
    vector<float> ans;
    for(i=0; i<length; i++){
        ans.push_back(data[i]);
    }
    return(ans);
}*/



MatStruct::MatStruct(){
    MatCount++;
    vector<int> tempDim;
    data = NULL;
    tempDim.push_back(0);
    length = 0;
    setDim(tempDim);
}

MatStruct::MatStruct(int dim1){
    MatCount++;
    length = 1;
    dimension.push_back(dim1);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

MatStruct::MatStruct(int dim1, int dim2){
    MatCount++;
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

MatStruct::MatStruct(int dim1, int dim2, int dim3){
    MatCount++;
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    dimension.push_back(dim3);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

MatStruct::MatStruct(int dim1, int dim2, int dim3, int dim4){
    MatCount++;
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    dimension.push_back(dim3);
    dimension.push_back(dim4);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

MatStruct::MatStruct(int dim1, int dim2, int dim3, int dim4, int dim5){
    MatCount++;
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    dimension.push_back(dim3);
    dimension.push_back(dim4);
    dimension.push_back(dim5);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

MatStruct::MatStruct(vector<int> dim){
    MatCount++;
    length = 1;
    dimension = dim;
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new float[length];
}

MatStruct::MatStruct(const MatStruct &inMat){
    MatCount++;
    //cout << "MatStruct::MatStruct() copy constructor called.\n";
    this->length = inMat.length;
    this->dimension = inMat.dimension;
    this->data = new float[inMat.length];
    for(i=0; i<inMat.length; i++){
        this->data[i] = inMat.data[i];
    }
}

MatStruct::~MatStruct(){
    MatCount--;
    //cout << "MatStruct::~MatStruct() called.\n";
    if(data != NULL){
        delete[] data;
    }
}

MatStruct& MatStruct::operator = (const MatStruct& inMat){
    //cout << "MatStruct::operator =  called.\n";
    if(this->length != inMat.length){
        if(this->data != NULL){
            delete[] this->data;
        }
        this->data = new float[inMat.length];
        this->length = inMat.length;
    }
    this->dimension = inMat.dimension;
    for(i=0; i<inMat.length; i++){
        this->data[i] = inMat.data[i];
    }
    return(*this);
}

vector<vector<int> > MatStruct::find(string oper, float val){
    vector<vector<int> > coor;
    if(oper == "=="){
        for(i=0; i<length; i++){
            if(data[i] == val){
                coor.push_back(linear2multiD(i));
            }
        }
    }else if(oper == "<"){
        for(i=0; i<length; i++){
            if(data[i] < val){
                coor.push_back(linear2multiD(i));
            }
        }
    }else if(oper == ">"){
        for(i=0; i<length; i++){
            if(data[i] > val){
                coor.push_back(linear2multiD(i));
            }
        }
    }
    //MatStruct ans(coor.size(), this->dimension.size());
    if(coor.size() > 0){
        /*
        for(i=0; i<this->dimension.size(); i++){
            for(j=0; j<coor.size(); j++){
                ans.set(coor[j][i], i*coor.size() + j);
            }
        }
        */
    }else{
        cout << "MatStruct::find() No match results.\n";
    }
    //cout << "MatStruct::find ans address: " << &ans << endl;
    return(coor);
}

vector<int> MatStruct::size(){
    return(dimension);
}

int MatStruct::size(int askDim){
    int ans = 0;
    if(askDim < dimension.size()){
        ans = dimension[askDim];
    }else{
        cout << "MatStruct::size() The requested dimension exceeds the dimension of this matrix.\n";
    }
    return(ans);
}

long long int MatStruct::getLength(){
    return(length);
}

bool MatStruct::checkBusted(vector<int> coor){
    bool ans = true;
    for(i=0; i<coor.size(); i++){
        if(coor[i] >= dimension[i]){
            ans = false;
            cout << "MatStruct::checkBusted(): dimension out of range.\n";
            cout << "in coor size: ";
            for(int i2=0; i2<coor.size(); i2++){
                cout << coor[i2] << " ";
            }
            cout << endl;
            
            cout << "matrix size: ";
            for(int i2=0; i2<dimension.size(); i2++){
                cout << dimension[i2] << " ";
            }
            cout << endl;
            throw runtime_error("MatStruct::checkBusted(): runtime error.");
            break;
        }
    }
    return(ans);
}

bool MatStruct::checkBusted(int dim1){
    bool ans = true;
    if(dimension[0] <= dim1){
        ans = false;
        cout << "MatStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatStruct::checkBusted(int dim1, int dim2){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2)){
        ans = false;
        cout << "MatStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatStruct::checkBusted(int dim1, int dim2, int dim3){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2) || (dimension[2] <= dim3)){
        ans = false;
        cout << "MatStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatStruct::checkBusted(int dim1, int dim2, int dim3, int dim4){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2) || (dimension[2] <= dim3) || (dimension[3] <= dim4)){
        ans = false;
        cout << "MatStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatStruct::checkBusted(int dim1, int dim2, int dim3, int dim4, int dim5){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2) || (dimension[2] <= dim3) || (dimension[3] <= dim4) || (dimension[4] <= dim5)){
        ans = false;
        cout << "MatStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatStruct::checkBusted(): runtime error.");
    }
    return(ans);
}


void MatStruct::setAll(float val){
    for(i=0; i<length; i++){
        data[i] = val;
    }
}

void MatStruct::setCertain(float val, vector<int> coor1, vector<int> coor2){
    if((coor1.size() != dimension.size()) || (coor2.size() != dimension.size())){
        cout << "MatStruct::setCertain(): dimension not match!\n";
        throw runtime_error("MatStruct::setCertain(): runtime error.");
    }else{for(i=0; i<coor1.size(); i++){
            if((coor1[i] == -1) || (coor2[i] == -1)){
                coor1[i] = 0;
                coor2[i] = dimension[i]-1;
            }
        }
        if(checkBusted(coor1) && checkBusted(coor2)){
            if(coor1.size() == 1){
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    data[i] = val;
                }
            }else if(coor1.size() == 2){
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        data[i + j*dimension[0]] = val;
                    }
                }
            }else if(coor1.size() == 3){
                int temp1 = dimension[0]*dimension[1];
            
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        for(k=coor1[2]; k<coor2[2]+1; k++){
                            data[i + j*dimension[0] + k*temp1] = val;
                        }
                    }
                }
            }else if(coor1.size() == 4){
                int temp1 = dimension[0]*dimension[1];
                int temp2 = dimension[0]*dimension[1]*dimension[2];
            
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        for(k=coor1[2]; k<coor2[2]+1; k++){
                            for(m=coor1[3]; m<coor2[3]+1; m++){
                                data[i + j*dimension[0] + k*temp1 + m*temp2] = val;
                            }
                        }
                    }
                }
            }else if(coor1.size() == 5){
                int temp1 = dimension[0]*dimension[1];
                int temp2 = dimension[0]*dimension[1]*dimension[2];
                int temp3 = dimension[0]*dimension[1]*dimension[2]*dimension[3];
            
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        for(k=coor1[2]; k<coor2[2]+1; k++){
                            for(m=coor1[3]; m<coor2[3]+1; m++){
                                for(n=coor1[4]; n<coor2[4]+1; n++){
                                    data[i + j*dimension[0] + k*temp1 + m*temp2 + n*temp3] = val;
                                }
                            }
                        }
                    }
                }
            }else{
                cout << "MatStruct::setCertain(): dimension too high to do this.\n";
                throw runtime_error("MatStruct::setCertain(): runtime error.");
            }
        }
    }
}

void MatStruct::set(float val, long long int dim1){
    if(dim1 < length){
        data[dim1] = val;
    }else{
        cout << "MatStruct::set(): dimension out of range.\n";
        throw runtime_error("MatStruct::set(): runtime error.");
    }
}

void MatStruct::set(float val, int dim1, int dim2){
    if(dimension.size() == 2){
        if(checkBusted(dim1, dim2))
            data[dim1 + dim2*dimension[0]] = val;
    }else{
        cout << "MatStruct::set(): dimension should be 2.\n";
        throw runtime_error("MatStruct::set(): runtime error.");
    }
}

void MatStruct::set(float val, int dim1, int dim2, int dim3){
    if(dimension.size() == 3){
        if(checkBusted(dim1, dim2, dim3))
            data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1]] = val;
    }else{
        cout << "MatStruct::set(): dimension should be 3.\n";
        throw runtime_error("MatStruct::set(): runtime error.");
    }
}

void MatStruct::set(float val, int dim1, int dim2, int dim3, int dim4){
    if(dimension.size() == 4){
        if(checkBusted(dim1, dim2, dim3, dim4))
            data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2]] = val;
    }else{
        cout << "MatStruct::set(): dimension should be 4.\n";
        throw runtime_error("MatStruct::set(): runtime error.");
    }
}

void MatStruct::set(float val, int dim1, int dim2, int dim3, int dim4, int dim5){
    if(dimension.size() == 5){
        if(checkBusted(dim1, dim2, dim3, dim4, dim5))
            data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2] + dim5*dimension[0]*dimension[1]*dimension[2]*dimension[3]] = val;
    }else{
        cout << "MatStruct::set(): dimension should be 5.\n";
        throw runtime_error("MatStruct::set(): runtime error.");
    }
}

void MatStruct::set(float val, vector<int> coor){
    if(coor.size() == dimension.size()){
        if(checkBusted(coor)){
        if(coor.size() == 1){
            set(val, coor[0]);
        }else if(coor.size() == 2){
            set(val, coor[0], coor[1]);
        }else if(coor.size() == 3){
            set(val, coor[0], coor[1], coor[2]);
        }else if(coor.size() == 4){
            set(val, coor[0], coor[1], coor[2], coor[3]);
        }else if(coor.size() == 5){
            set(val, coor[0], coor[1], coor[2], coor[3], coor[4]);
        }else{
            cout << "MatStruct::set(): dimension too high to do this.\n";
            throw runtime_error("MatStruct::set(): runtime error.");
        }}
    }else{
        cout << "MatStruct::set(): dimension not consistent.\n";
        throw runtime_error("MatStruct::set(): runtime error.");
    }
}

void MatStruct::set(float* in_vector){
    if(data != NULL){
        delete[] data;
    }
    data = new float[length];
    memcpy(data, in_vector, sizeof(float)*length);
}


void MatStruct::randAssign(){// This function assign random values to data.
    if (data != NULL){
        delete[] data;
    }
    data = new float[length];
    for(i=0; i<length; i++){
        data[i] = (float)rand()/(float)RAND_MAX - 0.5;
    }
}

void MatStruct::zeroAssign(){
    if (data != NULL){
        delete[] data;
    }
    data = new float[length];
    for(i=0; i<length; i++){
        data[i] = 0.0;
    }
}



vector<int> MatStruct::linear2multiD(int index){
    vector<int> ans;
    if(index < length){
        for(i=0; i<dimension.size(); i++){
            ans.push_back(index % dimension[i]);
            index = (index - ans[i])/dimension[i];
        }
    }else{
        cout << "MatStruct::linear2multiD(): dimension exceeds.\n";
    }
    return(ans);
}

int MatStruct::multiD2linear(vector<int> coor){
    int ans = -1;
    if(coor.size() == dimension.size()){
        if(checkBusted(coor)){
            int temp = 1;
            ans = 0;
            for(i=0; i<coor.size(); i++){
                ans += coor[i]*temp;
                temp *= dimension[i];
            }
        }
    }else{
        cout << "MatStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatStruct::multiD2linear(int dim1, int dim2){
    int ans = -1;
    if(dimension.size() == 2){
        if(checkBusted(dim1, dim2)){
            ans = dim1 + dimension[0]*dim2;
        }
    }else{
        cout << "MatStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatStruct::multiD2linear(int dim1, int dim2, int dim3){
    int ans = -1;
    if(dimension.size() == 3){
        if(checkBusted(dim1, dim2, dim3)){
            ans = dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1];
        }
    }else{
        cout << "MatStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatStruct::multiD2linear(int dim1, int dim2, int dim3, int dim4){
    int ans = -1;
    if(dimension.size() == 4){
        if(checkBusted(dim1, dim2, dim3, dim4)){
            ans = dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2];
        }
    }else{
        cout << "MatStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatStruct::multiD2linear(int dim1, int dim2, int dim3, int dim4, int dim5){
    int ans = -1;
    if(dimension.size() == 5){
        if(checkBusted(dim1, dim2, dim3, dim4, dim5)){
            ans = dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2] + dim5*dimension[0]*dimension[1]*dimension[2]*dimension[3];
        }
    }else{
        cout << "MatStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

float MatStruct::get(long long int dim1){
    float ans = -1;
    if(dim1 < length)
    ans = data[dim1];
    return(ans);
}

float MatStruct::get(int dim1, int dim2){
    float ans = -1;
    if(dimension.size() == 2){
        if(checkBusted(dim1, dim2))
            ans = data[dim1 + dim2*dimension[0]];
    }else{
        cout << "MatStruct::get(): dimension should be 2.\n";
    }
    return(ans);
}

float MatStruct::get(int dim1, int dim2, int dim3){
    float ans = -1;
    if(dimension.size() == 3){
        if(checkBusted(dim1, dim2, dim3))
            ans = data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1]];
    }else{
        cout << "MatStruct::get(): dimension should be 3.\n";
    }
    return(ans);
}

float MatStruct::get(int dim1, int dim2, int dim3, int dim4){
    float ans = -1;
    if(dimension.size() == 4){
        if(checkBusted(dim1, dim2, dim3, dim4))
            ans = data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2]];
    }else{
        cout << "MatStruct::get(): dimension should be 4.\n";
    }
    return(ans);
}

float MatStruct::get(int dim1, int dim2, int dim3, int dim4, int dim5){
    float ans = -1;
    if(dimension.size() == 5){
        if(checkBusted(dim1, dim2, dim3, dim4, dim5))
            ans = data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2] + dim5*dimension[0]*dimension[1]*dimension[2]*dimension[3]];
    }else{
        cout << "MatStruct::get(): dimension should be 5.\n";
    }
    return(ans);
}

float MatStruct::get(vector<int> coor){
    float ans = -1;
    if(coor.size() == dimension.size()){
        if(checkBusted(coor)){
        if(coor.size() == 1){
            ans = get(coor[0]);
        }else if(coor.size() == 2){
            ans = get(coor[0], coor[1]);
        }else if(coor.size() == 3){
            ans = get(coor[0], coor[1], coor[2]);
        }else if(coor.size() == 4){
            ans = get(coor[0], coor[1], coor[2], coor[3]);
        }else if(coor.size() == 5){
            ans = get(coor[0], coor[1], coor[2], coor[3], coor[4]);
        }else{
            cout << "MatStruct::get(): dimension too high to do this.\n";
        }}
    }else{
        cout << "MatStruct::get(): dimension not consistent.\n";
    }
    return(ans);
}

MatStruct MatStruct::get(vector<int> coor1, vector<int> coor2){
    //cout << "MatStruct::get(): here 0\n";
    MatStruct ans;
    //cout << "MatStruct::get(): here 1\n";
    if((coor1.size() == dimension.size()) && (coor2.size() == dimension.size())){
        for(i=0; i<coor1.size(); i++){
            if((coor1[i] == -1) || (coor2[i] == -1)){
                coor1[i] = 0;
                coor2[i] = dimension[i]-1;
                //cout << "MatStruct::get(): coor2[i] " << coor2[i] << "\n";
            }
        }
        //cout << "MatStruct::get(): here 2\n";
        if(checkBusted(coor1) && checkBusted(coor2)){
            vector<int> tempSize;
            //cout << "MatStruct::get(): here 3\n";
            for(i=0; i<coor1.size(); i++){
                tempSize.push_back(coor2[i] - coor1[i] + 1);
            }
            ans.setDim(tempSize);
            
            if(coor1.size() == 1){
                for(i=coor1[0]; i<=coor2[0]; i++){
                    ans.set(data[i], i-coor1[0]);
                }
            }else if(coor1.size() == 2){
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        ans.set(data[i + j*dimension[0]], i-coor1[0], j-coor1[1]);
                    }
                }
            }else if(coor1.size() == 3){
                long long int temp1 = dimension[0]*dimension[1];
                
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        for(k=coor1[2]; k<=coor2[2]; k++){
                            ans.set(data[i + j*dimension[0] + k*temp1], i-coor1[0], j-coor1[1], k-coor1[2]);
                        }
                    }
                }
            }else if(coor1.size() == 4){
                long long int temp1 = dimension[0]*dimension[1];
                long long int temp2 = dimension[0]*dimension[1]*dimension[2];
                
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        for(k=coor1[2]; k<=coor2[2]; k++){
                            for(m=coor1[3]; m<=coor2[3]; m++){
                                ans.set(data[i + j*dimension[0] + k*temp1 + m*temp2], i-coor1[0], j-coor1[1], k-coor1[2], m-coor1[3]);
                            }
                        }
                    }
                }
            }else if(coor1.size() == 5){
                long long int temp1 = dimension[0]*dimension[1];
                long long int temp2 = dimension[0]*dimension[1]*dimension[2];
                long long int temp3 = dimension[0]*dimension[1]*dimension[2]*dimension[3];

                //cout << "MatStruct::get(): here 4\n";
                
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        for(k=coor1[2]; k<=coor2[2]; k++){
                            for(m=coor1[3]; m<=coor2[3]; m++){
                                for(n=coor1[4]; n<=coor2[4]; n++){
                                    ans.set(data[i + j*dimension[0] + k*temp1 + m*temp2 + n*temp3], i-coor1[0], j-coor1[1], k-coor1[2], m-coor1[3], n-coor1[4]);
                                }
                            }
                        }
                    }
                }

                //cout << "MatStruct::get(): here 1\n";
            }else{
                cout << "MatStruct::get(): dimension too high to do this.\n";
            }
        }
    }else{
        cout << "MatStruct::get(): dimension not match!\n";
    }
    return(ans);
}

float* MatStruct::get(){
    float* ans = new float[length];
    for(i=0; i<length; i++){
        ans[i] = data[i];
    }
    return(ans);
}

int MatStruct::embed(MatStruct A, vector<int> beginConer){
    int ans = 1;
    MatStruct* subMatrix = &A;
    if(beginConer.size() == dimension.size()){
        vector<int> subSize = subMatrix->size();
        vector<int> endConer = dimension;
        for(i=0; i<dimension.size(); i++){
            endConer[i] = beginConer[i] + subSize[i] - 1;
        }
        if(checkBusted(endConer)){
            if(dimension.size() == 1){
                for(i=0; i<subSize[0]; i++){
                    set(subMatrix->get(i), i+beginConer[0]);
                }
            }else if(dimension.size() == 2){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        set(subMatrix->get(i, j), i+beginConer[0], j+beginConer[1]);
                    }
                }
            }else if(dimension.size() == 3){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        for(k=0; k<subSize[2]; k++){
                            set(subMatrix->get(i, j, k), i+beginConer[0], j+beginConer[1], k+beginConer[2]);
                        }
                    }
                }
            }else if(dimension.size() == 4){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        for(k=0; k<subSize[2]; k++){
                            for(m=0; m<subSize[3]; m++){
                                set(subMatrix->get(i, j, k, m), i+beginConer[0], j+beginConer[1], k+beginConer[2], m+beginConer[3]);
                            }
                        }
                    }
                }
            }else if(dimension.size() == 5){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        for(k=0; k<subSize[2]; k++){
                            for(m=0; m<subSize[3]; m++){
                                for(n=0; n<subSize[4]; n++){
                                    set(subMatrix->get(i, j, k, m, n), i+beginConer[0], j+beginConer[1], k+beginConer[2], m+beginConer[3], n+beginConer[4]);
                                }
                            }
                        }
                    }
                }
            }else{
                cout << "MatStruct::embed(): dimension too high to do this.\n";
            }
        }else{
            ans = -1;
        }
    }else{
        cout << "MatStruct::embed(): dimension not match!\n";
        ans = -1;
    }
    return(ans);
}

void MatStruct::show(){
    if(dimension.size() == 1){
        for(i=0; i<dimension[0]; i++){
            cout << data[i] << endl;
        }
        cout << endl;
    }else if(dimension.size() == 2){
        for(i=0; i<dimension[0]; i++){
            for(j=0; j<dimension[1]; j++){
                cout << data[i + j*dimension[0]] << " ";
            }
            cout << endl;
        }
    }else if(dimension.size() == 3){
        for(k=0; k<dimension[2]; k++){
        for(i=0; i<dimension[0]; i++){
            for(j=0; j<dimension[1]; j++){
                cout << data[i + j*dimension[0] + k*dimension[0]*dimension[1]] << " ";
            }
            cout << endl;
        }
        cout << endl;
        }
    }else{
        cout << "MatStruct::show(): dimension too high to show.\n";
        showDim();
    }
}

void MatStruct::showDim(){
    cout << "dimension of this matrix: ";
    for(int i=0; i<dimension.size(); i++){
        cout << dimension[i] << " ";
    }
    cout << endl;
}

float* MatStruct::getDataPt(){
    return(data);
}

void MatStruct::save(string fileName){
    ofstream ofile(fileName.c_str(), ios::binary); 

    // Write out data.
    save(ofile);
    ofile.close();
    
}

int MatStruct::writeHeader(ofstream& ofile){
    int dimensionN = dimension.size();
    int* dim = new int[dimensionN];
    for(int i=0; i<dimensionN; i++){
        dim[i] = dimension[i];
    }
    ofile.write((char*) &dimensionN, sizeof(int));
    ofile.write((char*) dim, sizeof(int)*dimensionN);
    delete [] dim;
    return 1;
}

int MatStruct::writeContent(ofstream& ofile){
    ofile.write((char*) data, sizeof(float)*length);
    return(1);
}

int MatStruct::save(ofstream& ofile){
    writeHeader(ofile);
    writeContent(ofile);
    return(1);
}

void MatStruct::load(string fileName){
    ifstream ifile(fileName.c_str(), ios::binary); 
    load(ifile);
    ifile.close();
}

int MatStruct::load(ifstream& ifile){
    int dimensionN;
    int* dim = NULL;
    int i;

    this->length = 1;

    ifile.read((char*) &dimensionN, sizeof(int));
    dim = new int[dimensionN];

    ifile.read((char*) dim, sizeof(int)*dimensionN);
    for(i=0; i<dimensionN; i++){
        this->dimension.push_back(dim[i]);
        length *= dim[i];
    }

    if(this->data != NULL){
        delete [] this->data;
    }    
    this->data = new float[length];
    ifile.read((char*) this->data, sizeof(float)*length);
    return(1);
}

void MatStruct::reshape(vector<int> in_dimension){
    long long int in_length = 1;
    for(i=0; i<in_dimension.size(); i++){
        in_length *= in_dimension[i];
    }
    if(in_length == length){
        dimension = in_dimension;
    }else{
        cout << "MatStruct::reshape(): dimension not match.\n";
    }
}

void MatStruct::rotate(string x, string y){
    if(dimension.size() < 2){
        cout << "MatStruct::rotate() dimension size < 2.\n";
    }else{
        if((x == "R") && (y == "C")){
            // Do nothing.
        }else{
            vector<int> newDimension;
            float* newData = new float[length];
            int count = 0;
            newDimension.assign(dimension.size(), 0);
            int layerNum = 1;
            for(i=2; i<dimension.size(); i++){
                newDimension[i] = dimension[i];
                layerNum *= dimension[i];
            }

            if(x == "R"){
                newDimension[0] = dimension[0];
                newDimension[1] = dimension[1];
            }else if(x == "-R"){
                newDimension[0] = dimension[0];
                newDimension[1] = dimension[1];
            }else if(x == "C"){
                newDimension[1] = dimension[0];
                newDimension[0] = dimension[1];
            }else if(x == "-C"){
                newDimension[1] = dimension[0];
                newDimension[0] = dimension[1];
            }

            for(m=0; m<layerNum; m++){
                int offset = m*dimension[0]*dimension[1];
                if(x == "R"){
                    if(y == "-C"){
                        for(j=dimension[1]-1; j>=0; j--){
                            for(i=0; i<dimension[0]; i++){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }else if(x == "-R"){
                    if(y == "C"){
                        for(j=0; j<dimension[1]; j++){
                            for(i=dimension[0]-1; i>=0; i--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }else if(y == "-C"){
                        for(j=dimension[1]-1; j>=0; j--){
                            for(i=dimension[0]-1; i>=0; i--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }else if(x == "C"){
                    if(y == "R"){
                        for(i=0; i<dimension[0]; i++){
                            for(j=0; j<dimension[1]; j++){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }else if(y == "-R"){
                        for(i=dimension[0]-1; i>=0; i--){
                            for(j=0; j<dimension[1]; j++){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }else if(x == "-C"){
                    if(y == "R"){
                        for(i=0; i<dimension[0]; i++){
                            for(j=dimension[1]-1; j>=0; j--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }else if(y == "-R"){
                        for(i=dimension[0]-1; i>=0; i--){
                            for(j=dimension[1]-1; j>=0; j--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }
            }
            dimension = newDimension;
            delete[] data;
            data = newData;
        }
    }
}

void MatStruct::rotate(string x, string y, string z){
    if(dimension.size() < 3){
        cout << "MatStruct::rotate() dimension size < 3.\n";
    }else{
        if((x == "R") && (y == "C") && (z == "H")){
            // Do nothing.
        }else{
            vector<int> newDimension;
            float* newData = new float[length];
            int count = 0;
            newDimension.assign(dimension.size(), 0);
            int layerNum = 1;
            for(i=3; i<dimension.size(); i++){
                newDimension[i] = dimension[i];
                layerNum *= dimension[i];
            }

            if((x == "R") || (x == "-R")){
                newDimension[0] = dimension[0];
                if((y == "C") || y == "-C"){
                    newDimension[1] = dimension[1];
                    newDimension[2] = dimension[2];
                }else if((y == "H") || y == "-H"){
                    newDimension[1] = dimension[2];
                    newDimension[2] = dimension[1];
                }
            }else if((x == "C") || (x == "-C")){
                newDimension[0] = dimension[1];
                if((y == "R") || y == "-R"){
                    newDimension[1] = dimension[0];
                    newDimension[2] = dimension[2];
                }else if((y == "H") || y == "-H"){
                    newDimension[1] = dimension[2];
                    newDimension[2] = dimension[0];
                }
            }else if((x == "H") || (x == "-H")){
                newDimension[0] = dimension[2];
                if((y == "R") || y == "-R"){
                    newDimension[1] = dimension[0];
                    newDimension[2] = dimension[1];
                }else if((y == "C") || y == "-C"){
                    newDimension[1] = dimension[1];
                    newDimension[2] = dimension[0];
                }
            }

            for(m=0; m<layerNum; m++){
                int offset = m*dimension[0]*dimension[1]*dimension[2];
                if(x == "R"){
                    if(y == "-C"){
                        if(z == "H"){// R -C H
                            for(k=0; k<dimension[2]; k++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// R -C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "C"){
                        if(z == "-H"){// R C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=0; j<dimension[1]; j++){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "C"){// R H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// R H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "C"){// R -H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// R -H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "-R"){
                    if(y == "C"){
                        if(z == "H"){// -R C H
                            for(k=0; k<dimension[2]; k++){
                                for(j=0; j<dimension[1]; j++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -R C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=0; j<dimension[1]; j++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-C"){
                        if(z == "H"){// -R -C H
                            for(k=0; k<dimension[2]; k++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -R -C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "C"){// -R H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -R H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "C"){// -R -H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -R -H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "C"){
                    if(y == "R"){
                        if(z == "H"){// C R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// C R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "H"){// C -R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// C -R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "R"){// C H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// C H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "R"){// C -H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// C -H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "-C"){
                    if(y == "R"){
                        if(z == "H"){// -C R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -C R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "H"){// -C -R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -C -R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "R"){// -C H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -C H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "R"){// -C -H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -C -H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "H"){
                    if(y == "R"){
                        if(z == "C"){// H R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// H R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "C"){// H -R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// H -R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "C"){
                        if(z == "R"){// H C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// H C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-C"){
                        if(z == "R"){// H -C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// H -C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "-H"){
                    if(y == "R"){
                        if(z == "C"){// -H R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -H R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "C"){// -H -R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -H -R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "C"){
                        if(z == "R"){// -H C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -H C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-C"){
                        if(z == "R"){// -H -C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -H -C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            dimension = newDimension;
            delete[] data;
            data = newData;
        }
    }
}


int reshape(MatStruct* matrix, vector<int> dimension){
    int ans = 1;
    
    for(int i=0; i<dimension.size(); i++){
        ans *= dimension[i];
    }
    if(ans == matrix->getLength()){
        matrix->dimension = dimension;
        ans = 1;
    }else{
        ans = -1;
        cout << "reshape(): dimension not match.\n";
    }
    return(ans);
}

MatStruct* MatCpy(MatStruct* originalMat){
    MatStruct* ans;
    ans = new MatStruct(originalMat->size());
    memcpy(ans->data, originalMat->data, sizeof(float)*(originalMat->length));
    return(ans);
}

MatStruct MatFromFile(string fileName){
    ifstream ifile(fileName.c_str(), ios::binary); 
    int dimensionN;
    int* dimension = NULL;
    vector<int> forMatDim;
    long long int length = 1;
    int i;
    float* data = NULL;

    ifile.read((char*) &dimensionN, sizeof(int));
    dimension = new int[dimensionN];
//    cout << "MatFromFile() data dimension: " << dimensionN << endl;

    ifile.read((char*) dimension, sizeof(int)*dimensionN);
    for(i=0; i<dimensionN; i++){
//        cout << dimension[i] << " ";
        forMatDim.push_back(dimension[i]);
        length *= dimension[i];
    }
//    cout << endl;
    MatStruct ans(forMatDim);
    
    data = new float[length];
    ifile.read((char*) data, sizeof(float)*length);
    ans.set(data);
    ifile.close();
    delete[] data;
    delete[] dimension;
    //cout << "MatFromFile() ans address: " << &ans << endl;
    return(ans);
}

vector<int> linear2multiD(int index, vector<int> dimension){
    int i;
    vector<int> ans;
        for(i=0; i<dimension.size(); i++){
            ans.push_back(index % dimension[i]);
            index = (index - ans[i])/dimension[i];
        }
    return(ans);
}





















void MatIntStruct::setDim(vector<int> newDimension){
    if(data != NULL){
        delete[] data;
        data = NULL;
    }
    dimension = newDimension;
    length = 1;
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}



MatIntStruct::MatIntStruct(){
//cout << "MatIntStruct() default constructor called" << endl;
    vector<int> tempDim;
    data = NULL;
    tempDim.push_back(0);
    length = 0;
    setDim(tempDim);
}

MatIntStruct::MatIntStruct(int dim1){
    length = 1;
    dimension.push_back(dim1);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}

MatIntStruct::MatIntStruct(int dim1, int dim2){
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}

MatIntStruct::MatIntStruct(int dim1, int dim2, int dim3){
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    dimension.push_back(dim3);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}

MatIntStruct::MatIntStruct(int dim1, int dim2, int dim3, int dim4){
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    dimension.push_back(dim3);
    dimension.push_back(dim4);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}

MatIntStruct::MatIntStruct(int dim1, int dim2, int dim3, int dim4, int dim5){
    length = 1;
    dimension.push_back(dim1);
    dimension.push_back(dim2);
    dimension.push_back(dim3);
    dimension.push_back(dim4);
    dimension.push_back(dim5);
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}

MatIntStruct::MatIntStruct(vector<int> dim){
    length = 1;
    dimension = dim;
    for(i=0; i<dimension.size(); i++){
        length *= dimension[i];
    }
    data = new int[length];
}

MatIntStruct::MatIntStruct(const MatIntStruct &inMat){
//    cout << "MatIntStruct::MatIntStruct() copy constructor called.\n";
    this->length = inMat.length;
    this->dimension = inMat.dimension;
    this->data = new int[inMat.length];
    for(i=0; i<inMat.length; i++){
        this->data[i] = inMat.data[i];
    }
}

MatIntStruct::~MatIntStruct(){
    if(data != NULL){
        delete[] data;
    }
    //cout << "MatIntStruct::~MatIntStruct() called.\n";
}

MatIntStruct& MatIntStruct::operator = (const MatIntStruct& inMat){
//    cout << "MatIntStruct::operator =  called.\n";
// cout << "MatIntStruct::operator =  inMat " << inMat.length << endl;
// cout << "MatIntStruct::operator =  this->length " << this->length << endl;
// cout << "MatIntStruct::operator =  inMat " << inMat.length << endl;
    if(this->length != inMat.length){
        if(this->data != NULL){
            delete[] this->data;
        }
        this->data = new int[inMat.length];
        this->length = inMat.length;
    }
    this->dimension = inMat.dimension;
    for(i=0; i<inMat.length; i++){
        this->data[i] = inMat.data[i];
    }
    return(*this);
}

vector<vector<int> > MatIntStruct::find(string oper, int val){
    vector<vector<int> > coor;
    if(oper == "=="){
        for(int i=0; i<length; i++){
            if(data[i] == val){
                coor.push_back(linear2multiD(i));
            }
        }
    }else if(oper == "<"){
        for(int i=0; i<length; i++){
            if(data[i] < val){
                coor.push_back(linear2multiD(i));
            }
        }
    }else if(oper == ">"){
        for(int i=0; i<length; i++){
            if(data[i] > val){
                coor.push_back(linear2multiD(i));
            }
        }
    }
    //MatIntStruct ans(coor.size(), this->dimension.size());
    if(coor.size() > 0){
        /*
        for(i=0; i<this->dimension.size(); i++){
            for(j=0; j<coor.size(); j++){
                ans.set(coor[j][i], i*coor.size() + j);
            }
        }
        */
    }else{
        cout << "MatIntStruct::find() No match results.\n";
    }
    //cout << "MatIntStruct::find ans address: " << &ans << endl;
    return(coor);
}

vector<int> MatIntStruct::size(){
    return(dimension);
}

int MatIntStruct::size(int askDim){
    int ans = 0;
    if(askDim < dimension.size()){
        ans = dimension[askDim];
    }else{
        cout << "MatIntStruct::size() The requested dimension exceeds the dimension of this matrix.\n";
    }
    return(ans);
}

long long int MatIntStruct::getLength(){
    return(length);
}

bool MatIntStruct::checkBusted(vector<int> coor){
    bool ans = true;
    for(i=0; i<coor.size(); i++){
        if(coor[i] >= dimension[i]){
            ans = false;
            cout << "MatIntStruct::checkBusted(): dimension out of range.\n";
            cout << "in coor size: ";
            for(int i2=0; i2<coor.size(); i2++){
                cout << coor[i2] << " ";
            }
            cout << endl;
            
            cout << "matrix size: ";
            for(int i2=0; i2<dimension.size(); i2++){
                cout << dimension[i2] << " ";
            }
            cout << endl;
            throw runtime_error("MatIntStruct::checkBusted(): runtime error.");
            break;
        }
    }
    return(ans);
}

bool MatIntStruct::checkBusted(int dim1){
    bool ans = true;
    if(dimension[0] <= dim1){
        ans = false;
        cout << "MatIntStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatIntStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatIntStruct::checkBusted(int dim1, int dim2){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2)){
        ans = false;
        cout << "MatIntStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatIntStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatIntStruct::checkBusted(int dim1, int dim2, int dim3){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2) || (dimension[2] <= dim3)){
        ans = false;
        cout << "MatIntStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatIntStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatIntStruct::checkBusted(int dim1, int dim2, int dim3, int dim4){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2) || (dimension[2] <= dim3) || (dimension[3] <= dim4)){
        ans = false;
        cout << "MatIntStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatIntStruct::checkBusted(): runtime error.");
    }
    return(ans);
}

bool MatIntStruct::checkBusted(int dim1, int dim2, int dim3, int dim4, int dim5){
    bool ans = true;
    if((dimension[0] <= dim1) || (dimension[1] <= dim2) || (dimension[2] <= dim3) || (dimension[3] <= dim4) || (dimension[4] <= dim5)){
        ans = false;
        cout << "MatIntStruct::checkBusted(): dimension out of range.\n";
        throw runtime_error("MatIntStruct::checkBusted(): runtime error.");
    }
    return(ans);
}


void MatIntStruct::setAll(int val){
    for(i=0; i<length; i++){
        data[i] = val;
    }
}

void MatIntStruct::setCertain(int val, vector<int> coor1, vector<int> coor2){
    if((coor1.size() != dimension.size()) || (coor2.size() != dimension.size())){
        cout << "MatIntStruct::setCertain(): dimension not match!\n";
    }else{for(i=0; i<coor1.size(); i++){
            if((coor1[i] == -1) || (coor2[i] == -1)){
                coor1[i] = 0;
                coor2[i] = dimension[i]-1;
            }
        }
        if(checkBusted(coor1) && checkBusted(coor2)){
            if(coor1.size() == 1){
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    data[i] = val;
                }
            }else if(coor1.size() == 2){
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        data[i + j*dimension[0]] = val;
                    }
                }
            }else if(coor1.size() == 3){
                int temp1 = dimension[0]*dimension[1];
            
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        for(k=coor1[2]; k<coor2[2]+1; k++){
                            data[i + j*dimension[0] + k*temp1] = val;
                        }
                    }
                }
            }else if(coor1.size() == 4){
                int temp1 = dimension[0]*dimension[1];
                int temp2 = dimension[0]*dimension[1]*dimension[2];
            
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        for(k=coor1[2]; k<coor2[2]+1; k++){
                            for(m=coor1[3]; m<coor2[3]+1; m++){
                                data[i + j*dimension[0] + k*temp1 + m*temp2] = val;
                            }
                        }
                    }
                }
            }else if(coor1.size() == 5){
                int temp1 = dimension[0]*dimension[1];
                int temp2 = dimension[0]*dimension[1]*dimension[2];
                int temp3 = dimension[0]*dimension[1]*dimension[2]*dimension[3];
            
                for(i=coor1[0]; i<coor2[0]+1; i++){
                    for(j=coor1[1]; j<coor2[1]+1; j++){
                        for(k=coor1[2]; k<coor2[2]+1; k++){
                            for(m=coor1[3]; m<coor2[3]+1; m++){
                                for(n=coor1[4]; n<coor2[4]+1; n++){
                                    data[i + j*dimension[0] + k*temp1 + m*temp2 + n*temp3] = val;
                                }
                            }
                        }
                    }
                }
            }else{
                cout << "MatIntStruct::setCertain(): dimension too high to do this.\n";
            }
        }
    }
}

void MatIntStruct::set(int val, long long int dim1){
    if(dim1 < length){
        data[dim1] = val;
    }else{
        cout << "MatIntStruct::set(): dimension out of range.\n";
    }
}

void MatIntStruct::set(int val, int dim1, int dim2){
    if(dimension.size() == 2){
        if(checkBusted(dim1, dim2))
            data[dim1 + dim2*dimension[0]] = val;
    }else{
        cout << "MatIntStruct::set(): dimension should be 2.\n";
    }
}

void MatIntStruct::set(int val, int dim1, int dim2, int dim3){
    if(dimension.size() == 3){
        if(checkBusted(dim1, dim2, dim3))
            data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1]] = val;
    }else{
        cout << "MatIntStruct::set(): dimension should be 3.\n";
    }
}

void MatIntStruct::set(int val, int dim1, int dim2, int dim3, int dim4){
    if(dimension.size() == 4){
        if(checkBusted(dim1, dim2, dim3, dim4))
            data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2]] = val;
    }else{
        cout << "MatIntStruct::set(): dimension should be 4.\n";
    }
}

void MatIntStruct::set(int val, int dim1, int dim2, int dim3, int dim4, int dim5){
    if(dimension.size() == 5){
        if(checkBusted(dim1, dim2, dim3, dim4, dim5))
            data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2] + dim5*dimension[0]*dimension[1]*dimension[2]*dimension[3]] = val;
    }else{
        cout << "MatIntStruct::set(): dimension should be 5.\n";
    }
}

void MatIntStruct::set(int val, vector<int> coor){
    if(coor.size() == dimension.size()){
        if(checkBusted(coor)){
        if(coor.size() == 1){
            set(val, coor[0]);
        }else if(coor.size() == 2){
            set(val, coor[0], coor[1]);
        }else if(coor.size() == 3){
            set(val, coor[0], coor[1], coor[2]);
        }else if(coor.size() == 4){
            set(val, coor[0], coor[1], coor[2], coor[3]);
        }else if(coor.size() == 5){
            set(val, coor[0], coor[1], coor[2], coor[3], coor[4]);
        }else{
            cout << "MatIntStruct::set(): dimension too high to do this.\n";
        }}
    }else{
        cout << "MatIntStruct::set(): dimension not consistent.\n";
    }
}

void MatIntStruct::set(int* in_vector){
    if(data != NULL){
        delete[] data;
    }
    data = new int[length];
    memcpy(data, in_vector, sizeof(int)*length);
}



vector<int> MatIntStruct::linear2multiD(int index){
    vector<int> ans;
    if(index < length){
        for(i=0; i<dimension.size(); i++){
            ans.push_back(index % dimension[i]);
            index = (index - ans[i])/dimension[i];
        }
    }else{
        cout << "MatIntStruct::linear2multiD(): dimension exceeds.\n";
    }
    return(ans);
}

int MatIntStruct::multiD2linear(vector<int> coor){
    int ans = -1;
    if(coor.size() == dimension.size()){
        if(checkBusted(coor)){
            int temp = 1;
            ans = 0;
            for(i=0; i<coor.size(); i++){
                ans += coor[i]*temp;
                temp *= dimension[i];
            }
        }
    }else{
        cout << "MatIntStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatIntStruct::multiD2linear(int dim1, int dim2){
    int ans = -1;
    if(dimension.size() == 2){
        if(checkBusted(dim1, dim2)){
            ans = dim1 + dimension[0]*dim2;
        }
    }else{
        cout << "MatIntStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatIntStruct::multiD2linear(int dim1, int dim2, int dim3){
    int ans = -1;
    if(dimension.size() == 3){
        if(checkBusted(dim1, dim2, dim3)){
            ans = dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1];
        }
    }else{
        cout << "MatIntStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatIntStruct::multiD2linear(int dim1, int dim2, int dim3, int dim4){
    int ans = -1;
    if(dimension.size() == 4){
        if(checkBusted(dim1, dim2, dim3, dim4)){
            ans = dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2];
        }
    }else{
        cout << "MatIntStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatIntStruct::multiD2linear(int dim1, int dim2, int dim3, int dim4, int dim5){
    int ans = -1;
    if(dimension.size() == 5){
        if(checkBusted(dim1, dim2, dim3, dim4, dim5)){
            ans = dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2] + dim5*dimension[0]*dimension[1]*dimension[2]*dimension[3];
        }
    }else{
        cout << "MatIntStruct::multiD2linear(): dimension not consistent.\n";
    }
    return(ans);
}

int MatIntStruct::get(long long int dim1){
    int ans = -1;
    if(dim1 < length)
    ans = data[dim1];
    return(ans);
}

int MatIntStruct::get(int dim1, int dim2){
    int ans = -1;
    if(dimension.size() == 2){
        if(checkBusted(dim1, dim2))
            ans = data[dim1 + dim2*dimension[0]];
    }else{
        cout << "MatIntStruct::get(): dimension should be 2.\n";
    }
    return(ans);
}

int MatIntStruct::get(int dim1, int dim2, int dim3){
    int ans = -1;
    if(dimension.size() == 3){
        if(checkBusted(dim1, dim2, dim3))
            ans = data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1]];
    }else{
        cout << "MatIntStruct::get(): dimension should be 3.\n";
    }
    return(ans);
}

int MatIntStruct::get(int dim1, int dim2, int dim3, int dim4){
    int ans = -1;
    if(dimension.size() == 4){
        if(checkBusted(dim1, dim2, dim3, dim4))
            ans = data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2]];
    }else{
        cout << "MatIntStruct::get(): dimension should be 4.\n";
    }
    return(ans);
}

int MatIntStruct::get(int dim1, int dim2, int dim3, int dim4, int dim5){
    int ans = -1;
    if(dimension.size() == 5){
        if(checkBusted(dim1, dim2, dim3, dim4, dim5))
            ans = data[dim1 + dim2*dimension[0] + dim3*dimension[0]*dimension[1] + dim4*dimension[0]*dimension[1]*dimension[2] + dim5*dimension[0]*dimension[1]*dimension[2]*dimension[3]];
    }else{
        cout << "MatIntStruct::get(): dimension should be 5.\n";
    }
    return(ans);
}

int MatIntStruct::get(vector<int> coor){
    int ans = -1;
    if(coor.size() == dimension.size()){
        if(checkBusted(coor)){
        if(coor.size() == 1){
            ans = get(coor[0]);
        }else if(coor.size() == 2){
            ans = get(coor[0], coor[1]);
        }else if(coor.size() == 3){
            ans = get(coor[0], coor[1], coor[2]);
        }else if(coor.size() == 4){
            ans = get(coor[0], coor[1], coor[2], coor[3]);
        }else if(coor.size() == 5){
            ans = get(coor[0], coor[1], coor[2], coor[3], coor[4]);
        }else{
            cout << "MatIntStruct::get(): dimension too high to do this.\n";
        }}
    }else{
        cout << "MatIntStruct::get(): dimension not consistent.\n";
    }
    return(ans);
}

MatIntStruct MatIntStruct::get(vector<int> coor1, vector<int> coor2){
    MatIntStruct ans;
    if((coor1.size() == dimension.size()) && (coor2.size() == dimension.size())){
        for(i=0; i<coor1.size(); i++){
            if((coor1[i] == -1) || (coor2[i] == -1)){
                coor1[i] = 0;
                coor2[i] = dimension[i]-1;
            }
        }
        if(checkBusted(coor1) && checkBusted(coor2)){
            vector<int> tempSize;
            for(i=0; i<coor1.size(); i++){
                tempSize.push_back(coor2[i] - coor1[i] + 1);
            }
            ans.setDim(tempSize);
            
            if(coor1.size() == 1){
                for(i=coor1[0]; i<=coor2[0]; i++){
                    ans.set(data[i], i-coor1[0]);
                }
            }else if(coor1.size() == 2){
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        ans.set(data[i + j*dimension[0]], i-coor1[0], j-coor1[1]);
                    }
                }
            }else if(coor1.size() == 3){
                long long int temp1 = dimension[0]*dimension[1];
                
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        for(k=coor1[2]; k<=coor2[2]; k++){
                            ans.set(data[i + j*dimension[0] + k*temp1], i-coor1[0], j-coor1[1], k-coor1[2]);
                        }
                    }
                }
            }else if(coor1.size() == 4){
                long long int temp1 = dimension[0]*dimension[1];
                long long int temp2 = dimension[0]*dimension[1]*dimension[2];
                
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        for(k=coor1[2]; k<=coor2[2]; k++){
                            for(m=coor1[3]; m<=coor2[3]; m++){
                                ans.set(data[i + j*dimension[0] + k*temp1 + m*temp2], i-coor1[0], j-coor1[1], k-coor1[2], m-coor1[3]);
                            }
                        }
                    }
                }
            }else if(coor1.size() == 5){
                long long int temp1 = dimension[0]*dimension[1];
                long long int temp2 = dimension[0]*dimension[1]*dimension[2];
                long long int temp3 = dimension[0]*dimension[1]*dimension[2]*dimension[3];

                
                for(i=coor1[0]; i<=coor2[0]; i++){
                    for(j=coor1[1]; j<=coor2[1]; j++){
                        for(k=coor1[2]; k<=coor2[2]; k++){
                            for(m=coor1[3]; m<=coor2[3]; m++){
                                for(n=coor1[4]; n<=coor2[4]; n++){
                                    ans.set(data[i + j*dimension[0] + k*temp1 + m*temp2 + n*temp3], i-coor1[0], j-coor1[1], k-coor1[2], m-coor1[3], n-coor1[4]);
                                }
                            }
                        }
                    }
                }

            }else{
                cout << "MatIntStruct::get(): dimension too high to do this.\n";
            }
        }
    }else{
        cout << "MatIntStruct::get(): dimension not match!\n";
    }
    return(ans);
}

int* MatIntStruct::get(){
    int* ans = new int[length];
    for(i=0; i<length; i++){
        ans[i] = data[i];
    }
    return(ans);
}

int MatIntStruct::embed(MatIntStruct A, vector<int> beginConer){
    int ans = 1;
    MatIntStruct* subMatrix = &A;
    if(beginConer.size() == dimension.size()){
        vector<int> subSize = subMatrix->size();
        vector<int> endConer = dimension;
        for(i=0; i<dimension.size(); i++){
            endConer[i] = beginConer[i] + subSize[i] - 1;
        }
        if(checkBusted(endConer)){
            if(dimension.size() == 1){
                for(i=0; i<subSize[0]; i++){
                    set(subMatrix->get(i), i+beginConer[0]);
                }
            }else if(dimension.size() == 2){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        set(subMatrix->get(i, j), i+beginConer[0], j+beginConer[1]);
                    }
                }
            }else if(dimension.size() == 3){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        for(k=0; k<subSize[2]; k++){
                            set(subMatrix->get(i, j, k), i+beginConer[0], j+beginConer[1], k+beginConer[2]);
                        }
                    }
                }
            }else if(dimension.size() == 4){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        for(k=0; k<subSize[2]; k++){
                            for(m=0; m<subSize[3]; m++){
                                set(subMatrix->get(i, j, k, m), i+beginConer[0], j+beginConer[1], k+beginConer[2], m+beginConer[3]);
                            }
                        }
                    }
                }
            }else if(dimension.size() == 5){
                for(i=0; i<subSize[0]; i++){
                    for(j=0; j<subSize[1]; j++){
                        for(k=0; k<subSize[2]; k++){
                            for(m=0; m<subSize[3]; m++){
                                for(n=0; n<subSize[4]; n++){
                                    set(subMatrix->get(i, j, k, m, n), i+beginConer[0], j+beginConer[1], k+beginConer[2], m+beginConer[3], n+beginConer[4]);
                                }
                            }
                        }
                    }
                }
            }else{
                cout << "MatIntStruct::embed(): dimension too high to do this.\n";
            }
        }else{
            ans = -1;
        }
    }else{
        cout << "MatIntStruct::embed(): dimension not match!\n";
        ans = -1;
    }
    return(ans);
}

void MatIntStruct::show(){
    if(dimension.size() == 1){
        for(i=0; i<dimension[0]; i++){
            cout << data[i] << endl;
        }
        cout << endl;
    }else if(dimension.size() == 2){
        for(i=0; i<dimension[0]; i++){
            for(j=0; j<dimension[1]; j++){
                cout << data[i + j*dimension[0]] << " ";
            }
            cout << endl;
        }
    }else if(dimension.size() == 3){
        for(k=0; k<dimension[2]; k++){
        for(i=0; i<dimension[0]; i++){
            for(j=0; j<dimension[1]; j++){
                cout << data[i + j*dimension[0] + k*dimension[0]*dimension[1]] << " ";
            }
            cout << endl;
        }
        cout << endl;
        }
    }else{
        cout << "MatIntStruct::show(): dimension too high to show.\n";
        showDim();
    }
}

void MatIntStruct::showDim(){
    cout << "dimension of this matrix: ";
    for(int i=0; i<dimension.size(); i++){
        cout << dimension[i] << " ";
    }
    cout << endl;
}

int* MatIntStruct::getDataPt(){
    return(data);
}

void MatIntStruct::save(string fileName){
    ofstream ofile(fileName.c_str(), ios::binary); 
    int* temp;
    int dimensionN = dimension.size();
    int* dim = new int[dimensionN];
    for(int i=0; i<dimensionN; i++){
        dim[i] = dimension[i];
    }
    ofile.write((char*) &dimensionN, sizeof(int));
    ofile.write((char*) dim, sizeof(int)*dimensionN);
    ofile.write((char*) data, sizeof(int)*length);
    ofile.close();
    
}

void MatIntStruct::reshape(vector<int> in_dimension){
    int in_length = 1;
    for(int i=0; i<in_dimension.size(); i++){
        in_length *= in_dimension[i];
    }
    if(in_length == length){
        dimension = in_dimension;
    }else{
        cout << "MatIntStruct::reshape(): dimension not match.\n";
    }
}

void MatIntStruct::rotate(string x, string y){
    if(dimension.size() < 2){
        cout << "MatIntStruct::rotate() dimension size < 2.\n";
    }else{
        if((x == "R") && (y == "C")){
            // Do nothing.
        }else{
            vector<int> newDimension;
            int* newData = new int[length];
            int count = 0;
            newDimension.assign(dimension.size(), 0);
            int layerNum = 1;
            for(i=2; i<dimension.size(); i++){
                newDimension[i] = dimension[i];
                layerNum *= dimension[i];
            }

            if(x == "R"){
                newDimension[0] = dimension[0];
                newDimension[1] = dimension[1];
            }else if(x == "-R"){
                newDimension[0] = dimension[0];
                newDimension[1] = dimension[1];
            }else if(x == "C"){
                newDimension[1] = dimension[0];
                newDimension[0] = dimension[1];
            }else if(x == "-C"){
                newDimension[1] = dimension[0];
                newDimension[0] = dimension[1];
            }

            for(m=0; m<layerNum; m++){
                int offset = m*dimension[0]*dimension[1];
                if(x == "R"){
                    if(y == "-C"){
                        for(j=dimension[1]-1; j>=0; j--){
                            for(i=0; i<dimension[0]; i++){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }else if(x == "-R"){
                    if(y == "C"){
                        for(j=0; j<dimension[1]; j++){
                            for(i=dimension[0]-1; i>=0; i--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }else if(y == "-C"){
                        for(j=dimension[1]-1; j>=0; j--){
                            for(i=dimension[0]-1; i>=0; i--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }else if(x == "C"){
                    if(y == "R"){
                        for(i=0; i<dimension[0]; i++){
                            for(j=0; j<dimension[1]; j++){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }else if(y == "-R"){
                        for(i=dimension[0]-1; i>=0; i--){
                            for(j=0; j<dimension[1]; j++){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }else if(x == "-C"){
                    if(y == "R"){
                        for(i=0; i<dimension[0]; i++){
                            for(j=dimension[1]-1; j>=0; j--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }else if(y == "-R"){
                        for(i=dimension[0]-1; i>=0; i--){
                            for(j=dimension[1]-1; j>=0; j--){
                                newData[count] = data[i + j*dimension[0] + offset];
                                count++;
                            }
                        }
                    }
                }
            }
            dimension = newDimension;
            delete[] data;
            data = newData;
        }
    }
}

void MatIntStruct::zeroAssign(){
    if (data != NULL){
        delete[] data;
    }
    data = new int[length];
    for(i=0; i<length; i++){
        data[i] = 0;
    }
}

void MatIntStruct::rotate(string x, string y, string z){
    if(dimension.size() < 3){
        cout << "MatIntStruct::rotate() dimension size < 3.\n";
    }else{
        if((x == "R") && (y == "C") && (z == "H")){
            // Do nothing.
        }else{
            vector<int> newDimension;
            int* newData = new int[length];
            int count = 0;
            newDimension.assign(dimension.size(), 0);
            int layerNum = 1;
            for(i=3; i<dimension.size(); i++){
                newDimension[i] = dimension[i];
                layerNum *= dimension[i];
            }

            if((x == "R") || (x == "-R")){
                newDimension[0] = dimension[0];
                if((y == "C") || y == "-C"){
                    newDimension[1] = dimension[1];
                    newDimension[2] = dimension[2];
                }else if((y == "H") || y == "-H"){
                    newDimension[1] = dimension[2];
                    newDimension[2] = dimension[1];
                }
            }else if((x == "C") || (x == "-C")){
                newDimension[0] = dimension[1];
                if((y == "R") || y == "-R"){
                    newDimension[1] = dimension[0];
                    newDimension[2] = dimension[2];
                }else if((y == "H") || y == "-H"){
                    newDimension[1] = dimension[2];
                    newDimension[2] = dimension[0];
                }
            }else if((x == "H") || (x == "-H")){
                newDimension[0] = dimension[2];
                if((y == "R") || y == "-R"){
                    newDimension[1] = dimension[0];
                    newDimension[2] = dimension[1];
                }else if((y == "C") || y == "-C"){
                    newDimension[1] = dimension[1];
                    newDimension[2] = dimension[0];
                }
            }

            for(m=0; m<layerNum; m++){
                int offset = m*dimension[0]*dimension[1]*dimension[2];
                if(x == "R"){
                    if(y == "-C"){
                        if(z == "H"){// R -C H
                            for(k=0; k<dimension[2]; k++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// R -C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "C"){
                        if(z == "-H"){// R C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=0; j<dimension[1]; j++){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "C"){// R H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// R H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "C"){// R -H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// R -H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=0; i<dimension[0]; i++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "-R"){
                    if(y == "C"){
                        if(z == "H"){// -R C H
                            for(k=0; k<dimension[2]; k++){
                                for(j=0; j<dimension[1]; j++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -R C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=0; j<dimension[1]; j++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-C"){
                        if(z == "H"){// -R -C H
                            for(k=0; k<dimension[2]; k++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -R -C -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "C"){// -R H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -R H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=0; k<dimension[2]; k++){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "C"){// -R -H C
                            for(j=0; j<dimension[1]; j++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -R -H -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(i=dimension[0]-1; i>=0; i--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "C"){
                    if(y == "R"){
                        if(z == "H"){// C R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// C R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "H"){// C -R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// C -R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "R"){// C H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// C H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "R"){// C -H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// C -H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=0; j<dimension[1]; j++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "-C"){
                    if(y == "R"){
                        if(z == "H"){// -C R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -C R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=0; i<dimension[0]; i++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "H"){// -C -R H
                            for(k=0; k<dimension[2]; k++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-H"){// -C -R -H
                            for(k=dimension[2]-1; k>=0; k--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "H"){
                        if(z == "R"){// -C H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -C H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=0; k<dimension[2]; k++){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-H"){
                        if(z == "R"){// -C -H R
                            for(i=0; i<dimension[0]; i++){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -C -H -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(k=dimension[2]-1; k>=0; k--){
                                    for(j=dimension[1]-1; j>=0; j--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "H"){
                    if(y == "R"){
                        if(z == "C"){// H R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// H R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "C"){// H -R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// H -R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "C"){
                        if(z == "R"){// H C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// H C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-C"){
                        if(z == "R"){// H -C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// H -C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=0; k<dimension[2]; k++){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }else if(x == "-H"){
                    if(y == "R"){
                        if(z == "C"){// -H R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -H R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=0; i<dimension[0]; i++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-R"){
                        if(z == "C"){// -H -R C
                            for(j=0; j<dimension[1]; j++){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-C"){// -H -R -C
                            for(j=dimension[1]-1; j>=0; j--){
                                for(i=dimension[0]-1; i>=0; i--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "C"){
                        if(z == "R"){// -H C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -H C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=0; j<dimension[1]; j++){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }else if(y == "-C"){
                        if(z == "R"){// -H -C R
                            for(i=0; i<dimension[0]; i++){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }else if(z == "-R"){// -H -C -R
                            for(i=dimension[0]-1; i>=0; i--){
                                for(j=dimension[1]-1; j>=0; j--){
                                    for(k=dimension[2]-1; k>=0; k--){
                                        newData[count] = data[i + j*dimension[0] + k*dimension[0]*dimension[1] + offset];
                                        count++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            dimension = newDimension;
            delete[] data;
            data = newData;
        }
    }
}


int reshape(MatIntStruct* matrix, vector<int> dimension){
    int ans = 1;
    
    for(int i=0; i<dimension.size(); i++){
        ans *= dimension[i];
    }
    if(ans == matrix->getLength()){
        matrix->dimension = dimension;
        ans = 1;
    }else{
        ans = -1;
        cout << "reshape(): dimension not match.\n";
    }
    return(ans);
}

MatIntStruct* MatCpy(MatIntStruct* originalMat){
    MatIntStruct* ans;
    ans = new MatIntStruct(originalMat->size());
    memcpy(ans->data, originalMat->data, sizeof(int)*(originalMat->length));
    return(ans);
}

MatIntStruct MatIntFromFile(string fileName){
    ifstream ifile(fileName.c_str(), ios::binary); 



    int dimensionN;
    int* dimension = NULL;
    vector<int> forMatDim;
    long long int length = 1;
    int i;
    int* data = NULL;


    ifile.read((char*) &dimensionN, sizeof(int));
    dimension = new int[dimensionN];


    ifile.read((char*) dimension, sizeof(int)*dimensionN);
    for(i=0; i<dimensionN; i++){
        forMatDim.push_back(dimension[i]);
        length *= dimension[i];
    }
    MatIntStruct ans(forMatDim);

    data = new int[length];
    ifile.read((char*) data, sizeof(int)*length);
    ans.set(data);
    ifile.close();

    delete[] data;
    data = NULL;
    delete[] dimension;
    dimension = NULL;

    return(ans);
}


MatStruct MatInt2Float(MatIntStruct inMat){
    vector<int> dimvector = inMat.size();
    MatStruct ans(dimvector);
    long long int length = inMat.getLength();
    long long int i;
    
    for(i=0; i<length; i++){
        ans.set((float) inMat.get(i), i);
    }
    return(ans);
}























