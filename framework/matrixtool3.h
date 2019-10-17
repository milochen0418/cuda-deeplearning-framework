#ifndef MATRIXTOOL3
#define MATRIXTOOL3

//#include "kernel.h"
#include <map>
#include <vector>
#include <string>

using namespace std;

vector<int> v1(int);
vector<int> v2(int, int);
vector<int> v3(int, int, int);
vector<int> v4(int, int, int, int);
vector<int> v5(int, int, int, int, int);


class MatStruct{
    private:
    long long int length;
    long long int i, j, k, m, n;
    vector<int> dimension;
    float* data;
    bool checkBusted(vector<int>);
    bool checkBusted(int);
    bool checkBusted(int, int);
    bool checkBusted(int, int, int);
    bool checkBusted(int, int, int, int);
    bool checkBusted(int, int, int, int, int);
    
    public:
    static int MatCount;
    MatStruct();
    MatStruct(int);
    MatStruct(int, int);
    MatStruct(int, int, int);
    MatStruct(int, int, int, int);
    MatStruct(int, int, int, int, int);
    MatStruct(vector<int>);
    MatStruct(const MatStruct&);
    ~MatStruct();
    
    
    vector<int> size();// It returns dimension.
    int size(int dim);
    void setAll(float val); // Fill data with val.
    void setCertain(float val, vector<int> coor1, vector<int> coor2); // Fill block from coor1 to coor2 with val.
    void set(float val, long long int linearIndex); 
    void set(float val, int, int);
    void set(float val, int, int, int);
    void set(float val, int, int, int, int);
    void set(float val, int, int, int, int, int);
    void set(float val, vector<int>);
    void set(float* val);
    void setDim(vector<int>);

    void rotate(string, string);
    void rotate(string, string, string);
    
    void randAssign();
    void zeroAssign();

    vector<vector<int> > find(string, float);
    void reshape(vector<int>);
    
    vector<int> linear2multiD(int);
    int multiD2linear(vector<int>);
    int multiD2linear(int, int);
    int multiD2linear(int, int, int);
    int multiD2linear(int, int, int, int);
    int multiD2linear(int, int, int, int, int);
    

    long long int getLength();
    float get(long long int);
    float get(int, int);
    float get(int, int, int);
    float get(int, int, int, int);
    float get(int, int, int, int, int);
    float get(vector<int>);
    MatStruct get(vector<int>, vector<int>); // Return a submatrix.
    float* get();// This returns a new generated pointer to the data. So modification to it doesn't affect the content of this object.
    float* getDataPt();// This returns the pointer to this->data. So midification will affect the content of this object.
    
    //vector<float> toVector();

    int embed(MatStruct B, vector<int> beginConer); // embed B into this. Return 1 means success.
    
    friend int reshape(MatStruct*, vector<int>);
    friend MatStruct* MatCpy(MatStruct*);
    
    
    void show();
    void showDim();
    void save(string fileName); // this saves header and data content.
    void load(string fileName);

    int save(ofstream& ofile); 
    int writeHeader(ofstream& ofile);
    int writeContent(ofstream& ofile); // this only writes data content.
    int load(ifstream& ifile);

    MatStruct& operator = (const MatStruct&);
};
MatStruct MatFromFile(string);
vector<int> linear2multiD(int, vector<int>);



class MatIntStruct{
    public:
    long long int length;
    long long int i, j, k, m, n;
    vector<int> dimension;
    int* data;
    bool checkBusted(vector<int>);
    bool checkBusted(int);
    bool checkBusted(int, int);
    bool checkBusted(int, int, int);
    bool checkBusted(int, int, int, int);
    bool checkBusted(int, int, int, int, int);
    
    public:
    MatIntStruct();
    MatIntStruct(int);
    MatIntStruct(int, int);
    MatIntStruct(int, int, int);
    MatIntStruct(int, int, int, int);
    MatIntStruct(int, int, int, int, int);
    MatIntStruct(vector<int>);
    MatIntStruct(const MatIntStruct&);
    ~MatIntStruct();
    
    void zeroAssign();
    
    
    vector<int> size();// It returns dimension.
    int size(int dim);
    void setAll(int val); // Fill data with val.
    void setCertain(int val, vector<int> coor1, vector<int> coor2); // Fill block from coor1 to coor2 with val.
    void set(int val, long long int linearIndex); 
    void set(int val, int, int);
    void set(int val, int, int, int);
    void set(int val, int, int, int, int);
    void set(int val, int, int, int, int, int);
    void set(int val, vector<int>);
    void set(int* val);
    void setDim(vector<int>);

    void rotate(string, string);
    void rotate(string, string, string);

    vector<vector<int> > find(string, int);
    void reshape(vector<int>);
    
    vector<int> linear2multiD(int);
    int multiD2linear(vector<int>);
    int multiD2linear(int, int);
    int multiD2linear(int, int, int);
    int multiD2linear(int, int, int, int);
    int multiD2linear(int, int, int, int, int);
    

    long long int getLength();
    int get(long long int);
    int get(int, int);
    int get(int, int, int);
    int get(int, int, int, int);
    int get(int, int, int, int, int);
    int get(vector<int>);
    MatIntStruct get(vector<int>, vector<int>); // Return a submatrix.
    int* get();// This returns a new generated pointer to the data. So modification to it doesn't affect the content of this object.
    int* getDataPt();// This returns the pointer to this->data. So midification will affect the content of this object.
    
    
    int embed(MatIntStruct B, vector<int> beginConer); // embed B into this. Return 1 means success.
    
    friend int reshape(MatIntStruct*, vector<int>);
    friend MatIntStruct* MatCpy(MatIntStruct*);
    
    
    void show();
    void showDim();
    void save(string fileName);
    void load(string fileName);

    MatIntStruct& operator = (const MatIntStruct&);
};
MatIntStruct MatIntFromFile(string);
MatStruct MatInt2Float(MatIntStruct);

#endif
