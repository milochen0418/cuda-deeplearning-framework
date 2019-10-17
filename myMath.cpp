#include <vector>
#include "myMath.h"
#include <math.h>
#include <iostream> // for cout
#include <cstdlib> // for rand
#include <algorithm> // for random_shuffle

using namespace std;

void permutation(int n, int m, int* result){
    vector<int> all(n);
    int i;
    for(i=0; i<n; i++){
        all[i] = i;
    }
    
    random_shuffle(all.begin(), all.end());
    
    
    for(i=0; i<m; i++){
        result[i] = all[i];
    }
    
}

randBag::randBag(){
    permedIdx = NULL;
    count = 0;
    partialSize = 0;
    allSize = 0;
}

randBag::randBag(int n, int m){
    permedIdx = NULL;
    reset(n, m);
}

randBag::~randBag(){
    if(permedIdx != NULL){
        delete[] permedIdx;
    }
}

randBag::randBag(const randBag& rhs):
permedIdx(new int[rhs.partialSize]),
count(rhs.count),
partialSize(rhs.partialSize),
allSize(rhs.allSize)
{
    for(int i = 0; i < this->partialSize; i++){
        this->permedIdx[i] = rhs.permedIdx[i];
    }
}

randBag& randBag::operator=(const randBag& rhs){
    int *temp = new int[rhs.partialSize];
    this->partialSize = rhs.partialSize;
    this->count = rhs.count;
    this->allSize = rhs.allSize;
    for(int i = 0; i < this->partialSize; i++){
        temp[i] = rhs.permedIdx[i];
    }
    if(this->permedIdx != NULL){
        delete[] this->permedIdx;
    }
    for(int i = 0; i < this->partialSize; i++){
        this->permedIdx[i] = temp[i];
    }

    return *this;
}

void randBag::reset(int n, int m){
    if(permedIdx != NULL){
        delete[] permedIdx;
    }
    permedIdx = new int[m];
    permutation(n, m, permedIdx);
    count = 0;
    partialSize = m;
    allSize = n;
}

int randBag::next(){
    if(partialSize == 0){
        cout << "randBag::next() reset this object first.\n";
        return(0);
    }else{
        int ans = permedIdx[count];
        count++;
        if(count == partialSize){
            reset(allSize, partialSize);
            cout << "randBag::next() index set regenerated.\n";
            count = 0;
        }
        return(ans);
    }
}

int randBag::soFar(){
    return(count);
}
