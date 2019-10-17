#ifndef DATATENSOR_
#define DATATENSOR_

#include <string>
#include "driver_types.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel3.h"
#include "matrixtool3.h"
#include <vector>

using namespace std;

// dataTensor stores data in GPU memory.
class dataTensor{
    private:
    int DL; // data length
    bool enabled;
    float* h_data;
    
    public:
    float* d_data;
    int* dimension; // This integer array is used to save the dimension of this dataTensor in host memory.
    int* d_dimension; // This integer array is used to save the dimension of this dataTensor in device memory.
    
    int dimensionN;
    int id;
    
    void setDataLength();
    int dataLength();
    void randAssign();
    void zeroAssign();
    void oneAssign();
    void resetData(float* data_pointer);
    void generalInit();
    dataTensor();
    dataTensor(int dimN, int* dim, float* data_pointer);
    dataTensor(int dimN, int* dim);
    dataTensor(int dim_1, float* data_pointer);
    dataTensor(int dim_1, int dim_2, float* data_pointer);
    dataTensor(int dim_1, int dim_2, int dim_3, float* data_pointer);
    dataTensor(int dim_1, int dim_2, int dim_3, int dim_4, float* data_pointer);
    dataTensor(int dim_1, int dim_2, int dim_3, int dim_4, int dim_5, float* data_pointer);
    dataTensor(int dim_1);
    dataTensor(int dim_1, int dim_2);
    dataTensor(int dim_1, int dim_2, int dim_3);
    dataTensor(int dim_1, int dim_2, int dim_3, int dim_4);
    dataTensor(int dim_1, int dim_2, int dim_3, int dim_4, int dim_5);
    ~dataTensor();
    float* getData();// This function returns a pointer to a new copy of the data in host.
    void showData();
    void showDim();
    bool getEnabled();
    void setTensor(int dimN, int* dim);
    void setDim(int dimN, int* dim);
    
    bool checkDimEqual(int dimN, int* dim); // returns true iff dimN == this->dimensionN and dim[i] == this->dimension[i]
    
    void save(string fileName);
    void load(string fileName);
    
    MatStruct getMatStruct();
};

void tensorSum(dataTensor* result, dataTensor* component, float c);// result += c*component, initialize result first.

#endif
