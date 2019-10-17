#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <math.h>
#include <stdlib.h>
#include "kernel3.h"
#include <limits>
#include "myMath.h"
//#include <time.h>

#define MAX2D_X_THREADS 16
#define MAX2D_Y_THREADS 32
#define MAX3D_X_THREADS 8
#define MAX3D_Y_THREADS 8
#define MAX3D_Z_THREADS 8
#define MAX_THREADS 512

#define CUDART_INF_F            __int_as_float(0x7f800000)

using namespace std;

int ALLBREAK = 0;

__global__ void GPUConv1Dv2(int* d_inputDim, float* d_input, int* d_weightDim, float* d_weight, int* d_outputDim, float* d_output, int* paddingN){
// Because this function is for one-dimensional convolution, 
// in_featureN should be d_weightDim[1] == d_inputDim[1], and
// out_featureN should be d_weightDim[2] == d_outputDim[1].
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_out_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]) && (idx_out_feature < d_outputDim[1])){
        int globalIdxOut = idx_out_feature*d_outputDim[0] + idx_output;
        d_output[globalIdxOut] = 0;
        for(int idx_in_feature=0; idx_in_feature<d_inputDim[1]; idx_in_feature++){
            int offsetWeight = idx_in_feature*d_weightDim[0] + idx_out_feature*d_weightDim[0]*d_weightDim[1];
            int offsetInput = idx_in_feature*d_inputDim[0];
            for(int k=0; k<d_weightDim[0]; k++){
                int localIdxX = idx_output-paddingN[0]+k;
                if((localIdxX >= 0) && (localIdxX < d_inputDim[0])){
                    d_output[globalIdxOut] += d_weight[offsetWeight + k]*d_input[offsetInput + localIdxX];
                }
            }
        }
    }
}

__global__ void GPUConv1Ddxv2(int* d_dedxDim, float* d_dedx, int* d_weightDim, float* d_weight, int* d_dedyDim, float* d_dedy, int* paddingN){
    int idx_dedx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedx < d_dedxDim[0]) && (idx_dedx_feature < d_dedxDim[1])){
        int globaldedxIdx = idx_dedx_feature*d_dedxDim[0] + idx_dedx;
        d_dedx[globaldedxIdx] = 0;
        for(int idx_dedy_feature=0; idx_dedy_feature<d_dedyDim[1]; idx_dedy_feature++){
            int offsetWeight = idx_dedy_feature*d_weightDim[0]*d_weightDim[1] + idx_dedx_feature*d_weightDim[0];
            int offsetdedy = idx_dedy_feature*d_dedyDim[0];
            for(int k=0; k<d_weightDim[0]; k++){
                int localIdxX = idx_dedx-paddingN[0]+k;
                if((localIdxX >= 0) && (localIdxX < d_dedyDim[0])){
                    d_dedx[globaldedxIdx] += d_weight[offsetWeight + d_weightDim[0] - k - 1]*d_dedy[offsetdedy + localIdxX];
                }
            }
        }
    }
}

__global__ void GPUConv1Ddwv2(int* d_inputDim, float* d_input, int* d_dedwDim, float* d_dedw, int* d_dedyDim, float* d_dedy, int* paddingN){
    int idx_dedw = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_input_feature = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_output_feature = blockIdx.z*blockDim.z + threadIdx.z;
    if((idx_dedw < d_dedwDim[0]) && (idx_input_feature < d_inputDim[1]) && (idx_output_feature < d_dedyDim[1])){
        int globaldedwIdx = idx_dedw + idx_input_feature*d_dedwDim[0] + idx_output_feature*d_dedwDim[0]*d_dedwDim[1];
        d_dedw[globaldedwIdx] = 0;
        int offsetInput = idx_input_feature*d_inputDim[0];
        int offsetdedy = idx_output_feature*d_dedyDim[0];
        for(int j=0; j<d_dedyDim[0]; j++){
            int localIdxX = idx_dedw-paddingN[0]+j;
            if((localIdxX >= 0) && (localIdxX < d_inputDim[0])){
                d_dedw[globaldedwIdx] += d_input[localIdxX + offsetInput]*d_dedy[j + offsetdedy];
            }
        }
    }
}



__global__ void GPUConv2Dv2(int* d_inputDim, float* d_input, int* d_weightDim, float* d_weight, int* d_outputDim, float* d_output, int* paddingN){
// Because this function is for two-dimensional convolution, 
// in_featureN should be d_weightDim[2] == d_inputDim[2], and
// out_featureN should be d_weightDim[3] == d_outputDim[2].
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_out_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]*d_outputDim[1]) && (idx_out_feature < d_outputDim[2])){
        int globalIdxOut = idx_out_feature*d_outputDim[0]*d_outputDim[1] + idx_output;
        d_output[globalIdxOut] = 0;
        
        int idx_output_x, idx_output_y, k1, k2;// idx_output_z;
        idx_output_x = idx_output % d_outputDim[0];
        idx_output_y = (idx_output - idx_output_x)/d_outputDim[0];
        //idx_output = (idx_output - idx_output_x)/d_outputDim[0];
        //idx_output_y = idx_output % d_outputDim[1];
        //idx_output = (idx_output - idx_output_y)/d_outputDim[1];
        
        
        for(int idx_in_feature=0; idx_in_feature<d_inputDim[2]; idx_in_feature++){
            int offsetWeight = idx_in_feature*d_weightDim[0]*d_weightDim[1] + idx_out_feature*d_weightDim[0]*d_weightDim[1]*d_weightDim[2];
            int offsetInput = idx_in_feature*d_inputDim[0]*d_inputDim[1];
            for(k1=0; k1<d_weightDim[0]; k1++){
                int localIdxX = idx_output_x-paddingN[0]+k1;
                if((localIdxX >= 0) && (localIdxX < d_inputDim[0])){
                    for(k2=0; k2<d_weightDim[1]; k2++){
                        int localIdxY = idx_output_y-paddingN[1]+k2;
                        if((localIdxY >= 0) && (localIdxY < d_inputDim[1])){
                            d_output[globalIdxOut] += d_weight[offsetWeight + k1 + k2*d_weightDim[0]]*d_input[offsetInput + localIdxX + localIdxY*d_inputDim[0]];
                        }
                    }
                }
            }
        }
    }
}

__global__ void GPUConv2Ddxv2(int* d_dedxDim, float* d_dedx, int* d_weightDim, float* d_weight, int* d_dedyDim, float* d_dedy, int* paddingN){
    int idx_dedx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedx < d_dedxDim[0]*d_dedxDim[1]) && (idx_dedx_feature < d_dedxDim[2])){
        int globaldedxIdx = idx_dedx_feature*d_dedxDim[0]*d_dedxDim[1] + idx_dedx;
        d_dedx[globaldedxIdx] = 0;
        
        int idx_dedx_x, idx_dedx_y, k1, k2;// idx_dedx_z;
        idx_dedx_x = idx_dedx % d_dedxDim[0];
        idx_dedx_y = (idx_dedx - idx_dedx_x)/d_dedxDim[0];
        //idx_dedx = (idx_dedx - idx_dedx_x)/d_dedxDim[0];
        //idx_dedx_y = idx_dedx % d_dedxDim[1];
        //idx_dedx = (idx_dedx - idx_dedx_y)/d_dedxDim[1];
        
        
        for(int idx_dedy_feature=0; idx_dedy_feature<d_dedyDim[2]; idx_dedy_feature++){
            int offsetWeight = idx_dedy_feature*d_weightDim[0]*d_weightDim[1]*d_weightDim[2] + idx_dedx_feature*d_weightDim[0]*d_weightDim[1];
            int offsetdedy = idx_dedy_feature*d_dedyDim[0]*d_dedyDim[1];
            for(k1=0; k1<d_weightDim[0]; k1++){
                int localIdxX = idx_dedx_x-paddingN[0]+k1;
                if((localIdxX >= 0) && (localIdxX < d_dedyDim[0])){
                    for(k2=0; k2<d_weightDim[1]; k2++){
                        int localIdxY = idx_dedx_y-paddingN[1]+k2;
                        if((localIdxY >= 0) && (localIdxY < d_dedyDim[1])){
                            d_dedx[globaldedxIdx] += d_weight[offsetWeight + d_weightDim[0] - k1 - 1 + (d_weightDim[1] - k2 - 1)*d_weightDim[0]]*d_dedy[offsetdedy + localIdxX + localIdxY*d_dedyDim[0]];
                        }
                    }
                }
            }
        }
    }
}

__global__ void GPUConv2Ddwv2(int* d_inputDim, float* d_input, int* d_dedwDim, float* d_dedw, int* d_dedyDim, float* d_dedy, int* paddingN){
    int idx_dedw = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_input_feature = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_output_feature = blockIdx.z*blockDim.z + threadIdx.z;
    if((idx_dedw < d_dedwDim[0]*d_dedwDim[1]) && (idx_input_feature < d_inputDim[2]) && (idx_output_feature < d_dedyDim[2])){
        int globaldedwIdx = idx_dedw + idx_input_feature*d_dedwDim[0]*d_dedwDim[1] + idx_output_feature*d_dedwDim[0]*d_dedwDim[1]*d_dedwDim[2];
        d_dedw[globaldedwIdx] = 0;
        
        int idx_dedw_x, idx_dedw_y, j1, j2;// idx_dedw_z;
        idx_dedw_x = idx_dedw % d_dedwDim[0];
        idx_dedw_y = (idx_dedw - idx_dedw_x)/d_dedwDim[0];
        //idx_dedw = (idx_dedw - idx_dedw_x)/d_dedwDim[0];
        //idx_dedw_y = idx_dedw % d_dedwDim[1];
        //idx_dedw = (idx_dedw - idx_dedw_y)/d_dedwDim[1];
        
        
        int offsetInput = idx_input_feature*d_inputDim[0]*d_inputDim[1];
        int offsetdedy = idx_output_feature*d_dedyDim[0]*d_dedyDim[1];
        for(j1=0; j1<d_dedyDim[0]; j1++){
            int localIdxX = idx_dedw_x-paddingN[0]+j1;
            if((localIdxX >= 0) && (localIdxX < d_inputDim[0])){
                for(j2=0; j2<d_dedyDim[1]; j2++){
                    int localIdxY = idx_dedw_y-paddingN[1]+j2;
                    if((localIdxY >= 0) && (localIdxY < d_inputDim[1])){
                        d_dedw[globaldedwIdx] += d_input[localIdxX + localIdxY*d_inputDim[0] + offsetInput]*d_dedy[j1 + j2*d_dedyDim[0] + offsetdedy];
                    }
                }
            }
        }
    }
}


__global__ void GPUConv3Dv2(int* d_inputDim, float* d_input, int* d_weightDim, float* d_weight, int* d_outputDim, float* d_output, int* paddingN){
// Because this function is for two-dimensional convolution, 
// in_featureN should be d_weightDim[3] == d_inputDim[3], and
// out_featureN should be d_weightDim[4] == d_outputDim[3].
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_out_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]*d_outputDim[1]*d_outputDim[2]) && (idx_out_feature < d_outputDim[3])){
        int globalIdxOut = idx_out_feature*d_outputDim[0]*d_outputDim[1]*d_outputDim[2] + idx_output;
        d_output[globalIdxOut] = 0;
        
        int idx_output_x, idx_output_y, idx_output_z, k1, k2, k3;// idx_output_z;
        idx_output_x = idx_output % d_outputDim[0];
        idx_output = (idx_output - idx_output_x)/d_outputDim[0];
        idx_output_y = idx_output % d_outputDim[1];
        idx_output_z = (idx_output - idx_output_y)/d_outputDim[1];
        //idx_output = (idx_output - idx_output_y)/d_outputDim[1];
        //idx_output_z = idx_output % d_outputDim[2];
        //idx_output = (idx_output - idx_output_y)/d_outputDim[2];
        
        
        for(int idx_in_feature=0; idx_in_feature<d_inputDim[3]; idx_in_feature++){
            int offsetWeight = idx_in_feature*d_weightDim[0]*d_weightDim[1]*d_weightDim[2] + idx_out_feature*d_weightDim[0]*d_weightDim[1]*d_weightDim[2]*d_weightDim[3];
            int offsetInput = idx_in_feature*d_inputDim[0]*d_inputDim[1]*d_inputDim[2];
            for(k1=0; k1<d_weightDim[0]; k1++){
                int localIdxX = idx_output_x-paddingN[0]+k1;
                if((localIdxX >= 0) && (localIdxX < d_inputDim[0])){
                    for(k2=0; k2<d_weightDim[1]; k2++){
                        int localIdxY = idx_output_y-paddingN[1]+k2;
                        if((localIdxY >= 0) && (localIdxY < d_inputDim[1])){
                            for(k3=0; k3<d_weightDim[2]; k3++){
                                int localIdxZ = idx_output_z-paddingN[2]+k3;
                                if((localIdxZ >= 0) && (localIdxZ < d_inputDim[2])){
                                    d_output[globalIdxOut] += d_weight[offsetWeight + k1 + k2*d_weightDim[0] + k3*d_weightDim[1]*d_weightDim[2]]*d_input[offsetInput + localIdxX + localIdxY*d_inputDim[0] + localIdxZ*d_inputDim[0]*d_inputDim[1]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


__global__ void GPUConv3Ddxv2(int* d_dedxDim, float* d_dedx, int* d_weightDim, float* d_weight, int* d_dedyDim, float* d_dedy, int* paddingN){
    int idx_dedx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedx < d_dedxDim[0]*d_dedxDim[1]*d_dedxDim[2]) && (idx_dedx_feature < d_dedxDim[3])){
        int globaldedxIdx = idx_dedx_feature*d_dedxDim[0]*d_dedxDim[1]*d_dedxDim[2] + idx_dedx;
        d_dedx[globaldedxIdx] = 0;
        
        int idx_dedx_x, idx_dedx_y, idx_dedx_z, k1, k2, k3;// idx_dedx_z;
        idx_dedx_x = idx_dedx % d_dedxDim[0];
        idx_dedx = (idx_dedx - idx_dedx_x)/d_dedxDim[0];
        idx_dedx_y = idx_dedx % d_dedxDim[1];
        idx_dedx_z = (idx_dedx - idx_dedx_y)/d_dedxDim[1];
        //idx_dedx = (idx_dedx - idx_dedx_y)/d_dedxDim[1];
        
        
        for(int idx_dedy_feature=0; idx_dedy_feature<d_dedyDim[3]; idx_dedy_feature++){
            int offsetWeight = idx_dedy_feature*d_weightDim[0]*d_weightDim[1]*d_weightDim[2]*d_weightDim[3] + idx_dedx_feature*d_weightDim[0]*d_weightDim[1]*d_weightDim[2];
            int offsetdedy = idx_dedy_feature*d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2];
            for(k1=0; k1<d_weightDim[0]; k1++){
                int localIdxX = idx_dedx_x-paddingN[0]+k1;
                if((localIdxX >= 0) && (localIdxX < d_dedyDim[0])){
                    for(k2=0; k2<d_weightDim[1]; k2++){
                        int localIdxY = idx_dedx_y-paddingN[1]+k2;
                        if((localIdxY >= 0) && (localIdxY < d_dedyDim[1])){
                            for(k3=0; k3<d_weightDim[2]; k3++){
                                int localIdxZ = idx_dedx_z-paddingN[2]+k3;
                                if((localIdxZ >= 0) && (localIdxZ < d_dedyDim[2])){
                                    d_dedx[globaldedxIdx] += d_weight[offsetWeight + d_weightDim[0] - k1 - 1 + (d_weightDim[1] - k2 - 1)*d_weightDim[0] + (d_weightDim[2] - k3 - 1)*d_weightDim[0]*d_weightDim[1]]*d_dedy[offsetdedy + localIdxX + localIdxY*d_dedyDim[0] + localIdxZ*d_dedyDim[0]*d_dedyDim[1]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}




__global__ void GPUConv3Ddwv2(int* d_inputDim, float* d_input, int* d_dedwDim, float* d_dedw, int* d_dedyDim, float* d_dedy, int* paddingN){
    int idx_dedw = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_input_feature = blockIdx.y*blockDim.y + threadIdx.y;
    int idx_output_feature = blockIdx.z*blockDim.z + threadIdx.z;
    if((idx_dedw < d_dedwDim[0]*d_dedwDim[1]*d_dedwDim[2]) && (idx_input_feature < d_inputDim[3]) && (idx_output_feature < d_dedyDim[3])){
        int globaldedwIdx = idx_dedw + idx_input_feature*d_dedwDim[0]*d_dedwDim[1]*d_dedwDim[2] + idx_output_feature*d_dedwDim[0]*d_dedwDim[1]*d_dedwDim[2]*d_dedwDim[3];
        d_dedw[globaldedwIdx] = 0;
        
        int idx_dedw_x, idx_dedw_y, idx_dedw_z, j1, j2, j3;// idx_dedw_z;
        idx_dedw_x = idx_dedw % d_dedwDim[0];
        idx_dedw = (idx_dedw - idx_dedw_x)/d_dedwDim[0];
        idx_dedw_y = idx_dedw % d_dedwDim[1];
        idx_dedw_z = (idx_dedw - idx_dedw_y)/d_dedwDim[1];
        
        
        int offsetInput = idx_input_feature*d_inputDim[0]*d_inputDim[1]*d_inputDim[2];
        int offsetdedy = idx_output_feature*d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2];
        for(j1=0; j1<d_dedyDim[0]; j1++){
            int localIdxX = idx_dedw_x-paddingN[0]+j1;
            if((localIdxX >= 0) && (localIdxX < d_inputDim[0])){
                for(j2=0; j2<d_dedyDim[1]; j2++){
                    int localIdxY = idx_dedw_y-paddingN[1]+j2;
                    if((localIdxY >= 0) && (localIdxY < d_inputDim[1])){
                        for(j3=0; j3<d_dedyDim[2]; j3++){
                            int localIdxZ = idx_dedw_z-paddingN[2]+j3;
                            if((localIdxZ >= 0) && (localIdxZ < d_inputDim[2])){
                                d_dedw[globaldedwIdx] += d_input[localIdxX + localIdxY*d_inputDim[0] + localIdxZ*d_inputDim[0]*d_inputDim[1] + offsetInput]*d_dedy[j1 + j2*d_dedyDim[0] + j3*d_dedyDim[0]*d_dedyDim[1] + offsetdedy];
                            }
                        }
                    }
                }
            }
        }
    }
}



__global__ void GPUBiasConv1Dv2(int* d_inputDim, float* d_input, float* d_bias, int* d_outputDim, float* d_output){
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_output_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]) && (idx_output_feature < d_outputDim[1])){
        int globalOutputIdx = idx_output + idx_output_feature*d_outputDim[0];
        d_output[globalOutputIdx] = d_input[globalOutputIdx] + d_bias[idx_output_feature];
    }
}


__global__ void GPUBiasConv2Dv2(int* d_inputDim, float* d_input, float* d_bias, int* d_outputDim, float* d_output){
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_output_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]*d_outputDim[1]) && (idx_output_feature < d_outputDim[2])){
        int globalOutputIdx = idx_output + idx_output_feature*d_outputDim[0]*d_outputDim[1];
        d_output[globalOutputIdx] = d_input[globalOutputIdx] + d_bias[idx_output_feature];
    }
}

__global__ void GPUBiasConv3Dv2(int* d_inputDim, float* d_input, float* d_bias, int* d_outputDim, float* d_output){
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_output_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]*d_outputDim[1]*d_outputDim[2]) && (idx_output_feature < d_outputDim[3])){
        int globalOutputIdx = idx_output + idx_output_feature*d_outputDim[0]*d_outputDim[1]*d_outputDim[2];
        d_output[globalOutputIdx] = d_input[globalOutputIdx] + d_bias[idx_output_feature];
    }
}


// dedw has to be set all zero before using this function.
__global__ void GPUConvBiasdedw1Dv2(float* d_dedw, int* d_dedyDim, float* d_dedy){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < d_dedyDim[0]) && (idx_dedy_feature < d_dedyDim[1])){
        atomicAdd(&d_dedw[idx_dedy_feature], d_dedy[idx_dedy + idx_dedy_feature*d_dedyDim[0]]);
    }
}


// dedw has to be set all zero before using this function.
__global__ void GPUConvBiasdedw2Dv2(float* d_dedw, int* d_dedyDim, float* d_dedy){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < d_dedyDim[0]*d_dedyDim[1]) && (idx_dedy_feature < d_dedyDim[2])){
        atomicAdd(&d_dedw[idx_dedy_feature], d_dedy[idx_dedy + idx_dedy_feature*d_dedyDim[0]*d_dedyDim[1]]);
    }
}


// dedw has to be set all zero before using this function.
__global__ void GPUConvBiasdedw3Dv2(float* d_dedw, int* d_dedyDim, float* d_dedy){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2]) && (idx_dedy_feature < d_dedyDim[3])){
        atomicAdd(&d_dedw[idx_dedy_feature], d_dedy[idx_dedy + idx_dedy_feature*d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2]]);
    }
}


// dedw has to be set all zero before using this function.
__global__ void GPUConvBiasdedwv2(float* d_dedw, float* d_dedy, int dataLength, int featureN){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < dataLength) && (idx_dedy_feature < featureN)){
        atomicAdd(&d_dedw[idx_dedy_feature], d_dedy[idx_dedy + idx_dedy_feature*dataLength]);
    }
}

__global__ void GPUZeroPad1Dv2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output, int x1)
{
    int idx_input = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_input < d_inputDim[0]) && (idx_feature < d_inputDim[1])){
        d_output[idx_input + idx_feature*d_outputDim[0] + x1] = d_input[idx_input + idx_feature*d_inputDim[0]];
    }
}

__global__ void GPUZeroPad2Dv2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output, int x1, int y1)
{
    int idx_input = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_input < d_inputDim[0]*d_inputDim[1]) && (idx_feature < d_inputDim[2])){
        int idx_input_x, idx_input_y, idx_output_x, idx_output_y;
        idx_input_x = idx_input % d_inputDim[0];
        idx_input_y = (idx_input - idx_input_x)/d_inputDim[0];
        
        idx_output_x = idx_input_x + x1;
        idx_output_y = idx_input_y + y1;
        
        d_output[idx_output_x + idx_output_y*d_outputDim[0] + idx_feature*d_outputDim[0]*d_outputDim[1]] = d_input[idx_input_x + idx_input_y*d_inputDim[0] + idx_feature*d_inputDim[0]*d_inputDim[1]];
    }
}

__global__ void GPUZeroPad3Dv2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output, int x1, int y1, int z1)
{

    int idx_input = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_input < d_inputDim[0]*d_inputDim[1]*d_inputDim[2]) && (idx_feature < d_inputDim[3])){
        int idx_input_x, idx_input_y, idx_input_z, idx_output_x, idx_output_y, idx_output_z;
        idx_input_x = idx_input % d_inputDim[0];
        idx_input = (idx_input - idx_input_x)/d_inputDim[0];
        idx_input_y = idx_input % d_inputDim[1];
        idx_input_z = (idx_input - idx_input_y)/d_inputDim[1];
        
        idx_output_x = idx_input_x + x1;
        idx_output_y = idx_input_y + y1;
        idx_output_z = idx_input_z + z1;
        
        d_output[idx_output_x + idx_output_y*d_outputDim[0] + idx_output_z*d_outputDim[0]*d_outputDim[1] + idx_feature*d_outputDim[0]*d_outputDim[1]*d_outputDim[2]] = d_input[idx_input_x + idx_input_y*d_inputDim[0] + idx_input_z*d_inputDim[0]*d_inputDim[1] + idx_feature*d_inputDim[0]*d_inputDim[1]*d_inputDim[2]];
    }
}



__global__ void GPUZeroPad1D(float* d_x, float* d_y, int x1, int dimx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < dimx){
        d_y[idx + x1] = d_x[idx];
    }
}

__global__ void GPUZeroPad2D(float* d_x, float* d_y, int x1, int x2, int y1, int dimx, int dimy)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx < dimx) && (idy < dimy)){
        d_y[(idy+y1)*(dimx + x1 + x2) + idx + x1] = d_x[idy*dimx + idx];
    }
}

__global__ void GPUZeroPad3D(float* d_x, float* d_y, int x1, int x2, int y1, int y2, int z1, int dimx, int dimy, int dimz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if((idx < dimx) && (idy < dimy) && (idz < dimz)){
        d_y[(idz + z1)*(dimx + x1 + x2)*(dimy + y1 + y2) + (idy+y1)*(dimx + x1 + x2) + idx + x1] = d_x[idz*dimx*dimy + idy*dimx + idx];
    }
}


__global__ void GPUZeroPadBack1Dv2(int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy, int x1)
{
    int idx_dedx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedx < d_dedxDim[0]) && (idx_feature < d_dedxDim[1])){
        d_dedx[idx_dedx + idx_feature*d_dedxDim[0]] = d_dedy[idx_dedx + idx_feature*d_dedyDim[0] + x1];
    }
}

__global__ void GPUZeroPadBack2Dv2(int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy, int x1, int y1)
{
    int idx_dedx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedx < d_dedxDim[0]*d_dedxDim[1]) && (idx_feature < d_dedxDim[2])){
        int idx_dedx_x, idx_dedx_y, idx_dedy_x, idx_dedy_y;
        idx_dedx_x = idx_dedx % d_dedxDim[0];
        idx_dedx_y = (idx_dedx - idx_dedx_x)/d_dedxDim[0];
        
        idx_dedy_x = idx_dedx_x + x1;
        idx_dedy_y = idx_dedx_y + y1;
        
        d_dedx[idx_dedx_x + idx_dedx_y*d_dedxDim[0] + idx_feature*d_dedxDim[0]*d_dedxDim[1]] = d_dedy[idx_dedy_x + idx_dedy_y*d_dedyDim[0] + idx_feature*d_dedyDim[0]*d_dedyDim[1]];
    }
}


__global__ void GPUZeroPadBack3Dv2(int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy, int x1, int y1, int z1)
{
    int idx_dedx = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedx < d_dedxDim[0]*d_dedxDim[1]*d_dedxDim[2]) && (idx_feature < d_dedxDim[3])){
        int idx_dedx_x, idx_dedx_y, idx_dedx_z, idx_dedy_x, idx_dedy_y, idx_dedy_z;
        idx_dedx_x = idx_dedx % d_dedxDim[0];
        idx_dedx = (idx_dedx - idx_dedx_x)/d_dedxDim[0];
        idx_dedx_y = idx_dedx % d_dedxDim[1];
        idx_dedx_z = (idx_dedx - idx_dedx_y)/d_dedxDim[1];
        
        idx_dedy_x = idx_dedx_x + x1;
        idx_dedy_y = idx_dedx_y + y1;
        idx_dedy_z = idx_dedx_z + z1;
        
        d_dedx[idx_dedx_x + idx_dedx_y*d_dedxDim[0] + idx_dedx_z*d_dedxDim[0]*d_dedxDim[1] + idx_feature*d_dedxDim[0]*d_dedxDim[1]*d_dedxDim[2]] = d_dedy[idx_dedy_x + idx_dedy_y*d_dedyDim[0] + idx_dedy_z*d_dedyDim[0]*d_dedyDim[1] + idx_feature*d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2]];
    }
}


__global__ void GPUZeroPadBack1D(float* d_dedy, float* d_dedx, int x1, int dimx)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < dimx){
        d_dedx[idx] = d_dedy[idx + x1];
    }
}

__global__ void GPUZeroPadBack2D(float* d_dedy, float* d_dedx, int x1, int x2, int y1, int dimx, int dimy)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx < dimx) && (idy < dimy)){
        d_dedx[idy*dimx + idx] = d_dedy[(idy+y1)*(dimx + x1 + x2) + idx + x1];
    }
}

__global__ void GPUZeroPadBack3D(float* d_dedy, float* d_dedx, int x1, int x2, int y1, int y2, int z1, int dimx, int dimy, int dimz)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int idy = blockIdx.y*blockDim.y + threadIdx.y;
    int idz = blockIdx.z*blockDim.z + threadIdx.z;
    if((idx < dimx) && (idy < dimy) && (idz < dimz)){
        d_dedx[idz*dimx*dimy + idy*dimx + idx] = d_dedy[(idz + z1)*(dimx + x1 + x2)*(dimy + y1 + y2) + (idy+y1)*(dimx + x1 + x2) + idx + x1];
    }
}

__global__ void GPUDiffSquare(float* d_in_x, float* d_tag, float* d_diff, float* d_out_y, 
                            int dataLength) // data length)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if ((idx < dataLength)){
        float temp = d_in_x[idx] - d_tag[idx];
        d_diff[idx] = temp;
        d_out_y[idx] = temp*temp;
    }
}

__global__ void GPUallAdd(float* d_x, float* d_y, int dataLength){
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if ((idx < dataLength)){
        atomicAdd(&d_y[0], d_x[idx]);
    }
}

__global__ void GPUDiffx2(float* d_in_x, float* d_tag, float* d_out_y, int dataLength) // data length)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if ((idx < dataLength)){
        float temp = d_in_x[idx] - d_tag[idx];
        d_out_y[idx] = 2*temp;
    }
}


__global__ void GPUMaxPoolConv2D(float* d_in_x, float* d_out_y, 
                            int ThC, int ThR, // thread number per col, row
                            int filterXL, int filterYL, // filter length. 
                            int xPadding, int yPadding, // padding number
                            int outXL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)){
        float ans = -CUDART_INF_F;
        float temp;

        int currentX, currentY;

        int loopx, loopy;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    if((currentY >= 0)&&(currentY < ThR)){
                        temp = d_in_x[currentX + currentY*ThC];
                        if(ans < temp){
                            ans = temp;
                        }
			            
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_out_y[blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + threadIdx.x + threadIdx.y*outXL] = ans;
    }
}

__global__ void GPUMaxPoolConv2Ddx(float* d_x, float* d_y, float* d_dedx, float* d_dedy,
                            int ThC, int ThR, // thread number per col, row
                            int filterXL, int filterYL, // filter length. 
                            int xPadding, int yPadding, // padding number
                            int outXL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)){
        float ans = 0;
        int index = blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + threadIdx.x + threadIdx.y*outXL;

        int currentX, currentY;

        int loopx, loopy;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    if((currentY >= 0)&&(currentY < ThR)){
                        if(d_y[currentX + currentY*ThC] == d_x[index]){
                            ans += d_dedy[currentX + currentY*ThC];
                        }
			            
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_dedx[index] = ans;
    }
}


__global__ void GPUMaxPoolConv3D(float* d_in_x, float* d_out_y, 
                            //int BLx, int BLy, int BLz, // block length
                            //int BNx, int BNy, int BNz, // block number
                            int ThC, int ThR, int ThD, // thread number per col, row, depth
                            int filterXL, int filterYL, int filterZL, // filter length. 
                            int xPadding, int yPadding, int zPadding, // padding number
                            int outXL, int outYL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)
      &&(blockIdx.z*blockDim.z + threadIdx.z < ThD)){
        float ans = -CUDART_INF_F;
        float temp;

        int currentX, currentY, currentZ;

        int loopx, loopy, loopz;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    currentZ = threadIdx.z - zPadding + blockIdx.z*blockDim.z;
                    if((currentY >= 0)&&(currentY < ThR)){
                        for (loopz=0; loopz<filterZL; loopz++){
                            if((currentZ >= 0)&&(currentZ < ThD)){
                                temp = d_in_x[currentX + currentY*ThC + currentZ*ThR*ThC];
                                if(ans < temp){
                                    ans = temp;
                                }
                            }
                            currentZ++;
                        }
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_out_y[blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + blockIdx.z*ThR*ThC*blockDim.z + threadIdx.x + threadIdx.y*outXL + threadIdx.z*outXL*outYL] = ans;
    }
}

__global__ void GPUMaxPoolConv3Ddx(float* d_x, float* d_y, float* d_dedx, float* d_dedy,
                            int ThC, int ThR, int ThD, // thread number per col, row, depth
                            int filterXL, int filterYL, int filterZL, // filter length. 
                            int xPadding, int yPadding, int zPadding, // padding number
                            int outXL, int outYL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)
      &&(blockIdx.z*blockDim.z + threadIdx.z < ThD)){
        float ans = 0;
        int index = blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + blockIdx.z*ThR*ThC*blockDim.z + threadIdx.x + threadIdx.y*outXL + threadIdx.z*outXL*outYL;

        int currentX, currentY, currentZ;

        int loopx, loopy, loopz;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    currentZ = threadIdx.z - zPadding + blockIdx.z*blockDim.z;
                    if((currentY >= 0)&&(currentY < ThR)){
                        for (loopz=0; loopz<filterZL; loopz++){
                            if((currentZ >= 0)&&(currentZ < ThD)){
                                //temp = d_in_x[currentX + currentY*ThC + currentZ*ThR*ThC];
                                if(d_y[currentX + currentY*ThC + currentZ*ThR*ThC] == d_x[index]){
                                    ans += d_dedy[currentX + currentY*ThC + currentZ*ThR*ThC];
                                }
                            }
                            currentZ++;
                        }
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_dedx[index] = ans;
    }
}


__global__ void GPUNormalBack(float* d_dedx, float* d_dedy, float* d_x, int maxInd, float maxVal, int length, int positive, float scale){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < length){
        float ans = 0.0;
        if(d_x[maxInd] == 0){
            ans = 1000*d_dedy[index];
        }else{
            if(index != maxInd){
                ans += (d_dedy[index]/maxVal)*scale;
            }else{
                for(int i=0; i<length; i++){
                    if(i != index){
                        ans -= d_x[i]*positive*d_dedy[i];
                    }
                }
                ans /= maxVal*maxVal;
                ans *= scale;
            }
            
        }
        d_dedx[index] = ans;
    }
}

__global__ void GPUConv3D(float* d_in_x, float* d_inFilter, float* d_out_y, 
                            //int BLx, int BLy, int BLz, // block length
                            //int BNx, int BNy, int BNz, // block number
                            int ThC, int ThR, int ThD, // thread number per col, row, depth
                            int filterXL, int filterYL, int filterZL, // filter length. 
                            int xPadding, int yPadding, int zPadding, // padding number
                            int outXL, int outYL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)
      &&(blockIdx.z*blockDim.z + threadIdx.z < ThD)){
        float ans = 0;

        int currentX, currentY, currentZ;

        int loopx, loopy, loopz;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    currentZ = threadIdx.z - zPadding + blockIdx.z*blockDim.z;
                    if((currentY >= 0)&&(currentY < ThR)){
                        for (loopz=0; loopz<filterZL; loopz++){
                            if((currentZ >= 0)&&(currentZ < ThD)){
                                ans += d_in_x[currentX + currentY*ThC + currentZ*ThR*ThC]*
                                        d_inFilter[loopx + loopy*filterXL + loopz*filterXL*filterYL];
                            }
                            currentZ++;
                        }
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_out_y[blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + blockIdx.z*ThR*ThC*blockDim.z + threadIdx.x + threadIdx.y*outXL + threadIdx.z*outXL*outYL] = ans;
    }
}

__global__ void GPUConv2D(float* d_in_x, float* d_inFilter, float* d_out_y, 
                            int ThC, int ThR, // thread number per col, row
                            int filterXL, int filterYL, // filter length. 
                            int xPadding, int yPadding, // padding number
                            int outXL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)){
        float ans = 0;

        int currentX, currentY;

        int loopx, loopy;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    if((currentY >= 0)&&(currentY < ThR)){
			ans += d_in_x[currentX + currentY*ThC]*d_inFilter[loopx + loopy*filterXL];
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_out_y[blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + threadIdx.x + threadIdx.y*outXL] = ans;
    }
}

// GPUConv3DBackdx is just a filter reversed convolution.
// d_in_dedy is de/dy. d_out_dedx is de/dx = sum de/dy * dy/dx.
__global__ void GPUConv3DBackdx(float* d_in_dedy, float* d_inFilter, float* d_out_dedx, 
                            //int BLx, int BLy, int BLz, // block length
                            //int BNx, int BNy, int BNz, // block number
                            int ThC, int ThR, int ThD, // thread number per col, row, depth
                            int filterXL, int filterYL, int filterZL, // filter length. 
                            int xPadding, int yPadding, int zPadding, // padding number
                            int outXL, int outYL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)
      &&(blockIdx.z*blockDim.z + threadIdx.z < ThD)){
        float ans = 0;

        int currentX, currentY, currentZ;

        int loopx, loopy, loopz;
        int filterL = filterXL*filterYL*filterZL - 1;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    currentZ = threadIdx.z - zPadding + blockIdx.z*blockDim.z;
                    if((currentY >= 0)&&(currentY < ThR)){
                        for (loopz=0; loopz<filterZL; loopz++){
                            if((currentZ >= 0)&&(currentZ < ThD)){
                                ans += d_in_dedy[currentX + currentY*ThC + currentZ*ThR*ThC]*
                                        d_inFilter[filterL - (loopx + loopy*filterXL + loopz*filterXL*filterYL)];
                            }
                            currentZ++;
                        }
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_out_dedx[blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + blockIdx.z*ThR*ThC*blockDim.z + threadIdx.x + threadIdx.y*outXL + threadIdx.z*outXL*outYL] = ans;
    }
}

// GPUConv2DBackdx is just a filter reversed convolution.
// d_in_dedy is de/dy. d_out_dedx is de/dx = sum de/dy * dy/dx.
__global__ void GPUConv2DBackdx(float* d_in_dedy, float* d_inFilter, float* d_out_dedx, 
                            int ThC, int ThR, // thread number per col, row
                            int filterXL, int filterYL, // filter length. 
                            int xPadding, int yPadding, // padding number
                            int outXL) // output data length
{
    if ((blockIdx.x*blockDim.x + threadIdx.x < ThC)
      &&(blockIdx.y*blockDim.y + threadIdx.y < ThR)){
        float ans = 0;

        int currentX, currentY;

        int loopx, loopy;
        int filterL = filterXL*filterYL - 1;
        currentX = threadIdx.x - xPadding + blockIdx.x*blockDim.x;
        for (loopx=0; loopx<filterXL; loopx++){
            currentY = threadIdx.y - yPadding + blockIdx.y*blockDim.y;
            if((currentX >= 0)&&(currentX < ThC)){
                for (loopy=0; loopy<filterYL; loopy++){
                    if((currentY >= 0)&&(currentY < ThR)){
                                ans += d_in_dedy[currentX + currentY*ThC]*
                                        d_inFilter[filterL - (loopx + loopy*filterXL)];
                    }
                    currentY++;
                }
            }
            currentX++;
        }
        d_out_dedx[blockIdx.x*blockDim.x + blockIdx.y*ThC*blockDim.y + threadIdx.x + threadIdx.y*outXL] = ans;
    }
}

// GPUConv3DBackdw has not been tested yet. d_in_dedy and d_out_dedw stand for de/dy and de/dw, respectively, where e is error. 
// This function does not adjust the weight of filter. Use GPUConvAdjustFilter to adjust the weight.
// If the upper bound of threadN/block is 512, then the bound of the dimension of filter is 7x7x7.
// If the upper bound of threadN/block is 1024, then the bound of the dimension of filter is 9x9x9.
// This has to be satisfied: filterXL*filterYL*filterZL <= threadN/block.
__global__ void GPUConv3DBackdw(float* d_x, float* d_dedy, float* d_dedw, int filterXL, int filterYL, int xDim0, int xDim1, int xDim2, int yDim0, int yDim1,
                                int xPadding, int yPadding, int zPadding){
    int xId = blockIdx.x+threadIdx.x-xPadding;
    int yId = blockIdx.y - yPadding + threadIdx.y;
    int zId = blockIdx.z - zPadding + threadIdx.z;
    if((xId >= 0) && (xId < xDim0) && (yId >= 0) && (yId < xDim1) && (zId >= 0) && (zId < xDim2)){
    atomicAdd(&d_dedw[threadIdx.x + threadIdx.y*filterXL + threadIdx.z*filterXL*filterYL], 
            d_x[xId + yId*xDim0 + zId*xDim0*xDim1]*d_dedy[blockIdx.x + blockIdx.y*yDim0 + blockIdx.z*yDim0*yDim1]);
    }
}

__global__ void GPUConv3DBackdw_old(float* d_in_x, float* d_in_dedy, float* d_out_dedw, int filterXL, int filterYL, 
                                int xPadding, int yPadding, int zPadding){
    // This function only can be used for the case when input data dimensions equal to output dimensions. This is the case if the
    // padding numbers are well-setted. For the not equal case, this function needs the input data dimensions as additional parameters.
    int xId = blockIdx.x - xPadding + threadIdx.x;// Block index is the index of y, while thread index is the index of weight. 
    int yId = blockIdx.y - yPadding + threadIdx.y;
    int zId = blockIdx.z - zPadding + threadIdx.z;
    if((xId >= 0)&&(xId < gridDim.x)&&
       (yId >= 0)&&(yId < gridDim.y)&&
       (zId >= 0)&&(zId < gridDim.z)){
           // de/dw = sum de/dy df/dw. d_in_x is dy/dw by computation.
           atomicAdd(&d_out_dedw[threadIdx.x + threadIdx.y*filterXL + threadIdx.z*filterXL*filterYL], 
           d_in_dedy[xId + yId*gridDim.x + zId*gridDim.x*gridDim.y]*d_in_x[xId + yId*gridDim.x + zId*gridDim.x*gridDim.y]);
       }
}

// GPUConv2DBackdw has not been tested yet. d_in_dedy and d_out_dedw stand for de/dy and de/dw, respectively, where e is error. 
// This function does not adjust the weight of filter. Use GPUConvAdjustFilter to adjust the weight.
// This has to be satisfied: filterXL*filterYL <= threadN/block.
__global__ void GPUConv2DBackdw(float* d_x, float* d_dedy, float* d_dedw, int filterXL, int xDim0, int xDim1, int yDim0,
                                int xPadding, int yPadding){
    int xId = blockIdx.x+threadIdx.x-xPadding;
    int yId = blockIdx.y - yPadding + threadIdx.y;
    if((xId >= 0) && (xId < xDim0) && (yId >= 0) && (yId < xDim1)){
    atomicAdd(&d_dedw[threadIdx.x + threadIdx.y*filterXL], 
            d_x[xId + yId*xDim0]*d_dedy[blockIdx.x + blockIdx.y*yDim0]);
    }
}

__global__ void GPUConv2DBackdw_old(float* d_in_x, float* d_in_dedy, float* d_out_dedw, int filterXL, 
                                int xPadding, int yPadding){
    // This function only can be used for the case when input data dimensions equal to output dimensions. This is the case if the
    // padding numbers are well-setted. For the not equal case, this function needs the input data dimensions as additional parameters.
    int xId = blockIdx.x - xPadding + threadIdx.x;// Block index is the index of y, while thread index is the index of weight. 
    int yId = blockIdx.y - yPadding + threadIdx.y;
    if((xId >= 0)&&(xId < gridDim.x)&&
       (yId >= 0)&&(yId < gridDim.y)){
           // de/dw = sum de/dy df/dw. d_in_x is dy/dw by computation.
           atomicAdd(&d_out_dedw[threadIdx.x + threadIdx.y*filterXL], 
           d_in_dedy[xId + yId*gridDim.x]*d_in_x[xId + yId*gridDim.x]);
       }
}

__global__ void GPUConvAdjustFilter(float alpha, float* d_filterWeight, float* d_dedw){
    d_filterWeight[threadIdx.x] -= alpha*d_dedw[threadIdx.x];// threadIdx.x is the index of the filter.
}

// Not tested yet, but had been checked by eyes.
__global__ void GPUBiasConv3D(float* d_in_x, float* d_out_y, float bias, int ThR, int ThC, int ThD){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;
    if((row < ThC)&&(col < ThR)&&(layer < ThD)){
        int index = row + col*ThC + layer*ThR*ThC;
        d_out_y[index] = d_in_x[index] + bias;
    }
}

// Not tested yet, but had been checked by eyes.
__global__ void GPUBiasConv2D(float* d_in_x, float* d_out_y, float bias, int ThR, int ThC){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if((row < ThC)&&(col < ThR)){
        int index = row + col*ThC;
        d_out_y[index] = d_in_x[index] + bias;
    }
}

// Not tested yet, but hardly be wrong.
// *d_outData should be initialized as 0 outside this function.
// This function can be used to computed de/dw for the bias after a convolution layer,
// with d_inData and *d_outData being de/dy and de/dw, respectively. (w is the weight of bias.)
__global__ void GPUAllSum(float* d_dedy, float* d_dedw, int ThR){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < ThR){
        atomicAdd(d_dedw, d_dedy[index]);
    }
}

__global__ void GPUConvAdjustBias(float alpha, float* d_biasWeight, float* d_dedw){
    d_biasWeight[threadIdx.x] -= alpha*d_dedw[threadIdx.x];// threadIdx.x is the index of the bias.
}

__global__ void GPUScale(float* d_in, float* d_out, float scale, int length){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    if(index < length){
        d_out[index] = d_in[index]*scale;
    }
}


__global__ void GPUMaxPooling2v2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output){
// Because this function is for one-dimensional data, 
// in_featureN should be d_inputDim[1], and
// out_featureN should be d_outputDim[1].
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_out_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]) && (idx_out_feature < d_outputDim[1])){
        int globalIdxOut = idx_out_feature*d_outputDim[0] + idx_output;
        int idx_input = idx_out_feature*d_inputDim[0] + 2*idx_output;
        if(d_input[idx_input] > d_input[idx_input + 1]){
            d_output[globalIdxOut] = d_input[idx_input];
        }else{
            d_output[globalIdxOut] = d_input[idx_input + 1];
        }
    }
}


//dedx should be assigned all zero before using this function.
__global__ void GPUMaxPoolingdx2v2(int* d_inputDim, float* d_input, int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < d_dedyDim[0]) && (idx_dedy_feature < d_dedyDim[1])){
        int idx_input = idx_dedy_feature*d_inputDim[0] + 2*idx_dedy;
        if(d_input[idx_input] > d_input[idx_input + 1]){
            d_dedx[idx_input] = d_dedy[idx_dedy_feature*d_dedyDim[0] + idx_dedy];
        }else{
            d_dedx[idx_input + 1] = d_dedy[idx_dedy_feature*d_dedyDim[0] + idx_dedy];
        }
    }
}


__global__ void GPUMaxPooling2x2v2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output){
// Because this function is for two-dimensional data, 
// in_featureN should be d_inputDim[2], and
// out_featureN should be d_outputDim[2].
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_out_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]*d_outputDim[1]) && (idx_out_feature < d_outputDim[2])){
        int globalIdxOut = idx_out_feature*d_outputDim[0]*d_outputDim[1] + idx_output;
        
        int idx_output_x, idx_output_y;// idx_output_z;
        idx_output_x = idx_output % d_outputDim[0];
        idx_output_y = (idx_output - idx_output_x)/d_outputDim[0];
        //idx_output = (idx_output - idx_output_x)/d_outputDim[0];
        //idx_output_y = idx_output % d_outputDim[1];
        //idx_output = (idx_output - idx_output_y)/d_outputDim[1];
        
        int offsetInput = idx_out_feature*d_inputDim[0]*d_inputDim[1];
        // (x, y): 2*idx_output_y*d_inputDim[0] + 2*idx_output_x + offsetInput;
        // (x+1, y): 2*idx_output_y*d_inputDim[0] + 2*idx_output_x + 1 + offsetInput;
        // (x, y+1): (2*idx_output_y + 1)*d_inputDim[0] + 2*idx_output_x + offsetInput;
        // (x+1, y+1): (2*idx_output_y + 1)*d_inputDim[0] + 2*idx_output_x + 1 + offsetInput;
        
        
        
        int idx_input = 2*idx_output_y*d_inputDim[0] + 2*idx_output_x + offsetInput;
        if(d_input[idx_input] > d_input[idx_input + 1]){
            d_output[globalIdxOut] = d_input[idx_input];//(x, y)
        }else{
            d_output[globalIdxOut] = d_input[idx_input + 1];//(x+1, y)
        }
        
        idx_input += d_inputDim[0];//(x, y+1)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
        
        idx_input += 1;//(x+1, y+1)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
    }
}


//dedx should be assigned all zero before using this function.
__global__ void GPUMaxPoolingdx2x2v2(int* d_inputDim, float* d_input, int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < d_dedyDim[0]*d_dedyDim[1]) && (idx_dedy_feature < d_dedyDim[2])){
    
        int idx_dedy_x, idx_dedy_y;// idx_dedy_z;
        idx_dedy_x = idx_dedy % d_dedyDim[0];
        idx_dedy_y = (idx_dedy - idx_dedy_x)/d_dedyDim[0];
        //idx_dedy = (idx_dedy - idx_dedy_x)/d_dedyDim[0];
        //idx_dedy_y = idx_dedy % d_dedyDim[1];
        //idx_dedy = (idx_dedy - idx_dedy_y)/d_dedyDim[1];
        
        int offsetInput = idx_dedy_feature*d_inputDim[0]*d_inputDim[1];
        // (x, y): 2*idx_dedy_y*d_inputDim[0] + 2*idx_dedy_x + offsetInput;
        // (x+1, y): 2*idx_dedy_y*d_inputDim[0] + 2*idx_dedy_x + 1 + offsetInput;
        // (x, y+1): (2*idx_dedy_y + 1)*d_inputDim[0] + 2*idx_dedy_x + offsetInput;
        // (x+1, y+1): (2*idx_dedy_y + 1)*d_inputDim[0] + 2*idx_dedy_x + 1 + offsetInput;
        
        
        int idx_dedx = 2*(idx_dedy_y*d_dedxDim[0] + idx_dedy_x) + offsetInput;
        int tempIdx;
        float tempVal;
        
        if(d_input[idx_dedx] > d_input[idx_dedx + 1]){
            tempIdx = idx_dedx; //(x, y)
            tempVal = d_input[idx_dedx];
        }else{
            tempIdx = idx_dedx + 1; //(x+1, y)
            tempVal = d_input[idx_dedx + 1];
        }
        
        idx_dedx += d_inputDim[0]; //(x, y+1)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        idx_dedx += 1; //(x+1, y+1)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        d_dedx[tempIdx] = d_dedy[idx_dedy_feature*d_dedyDim[0]*d_dedyDim[1] + idx_dedy];
    }
}


__global__ void GPUMaxPooling2x2x2v2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output){
// Because this function is for three-dimensional data, 
// in_featureN should be d_inputDim[3], and
// out_featureN should be d_outputDim[3].
    int idx_output = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_out_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_output < d_outputDim[0]*d_outputDim[1]*d_outputDim[2]) && (idx_out_feature < d_outputDim[3])){
        int globalIdxOut = idx_out_feature*d_outputDim[0]*d_outputDim[1]*d_outputDim[2] + idx_output;
        
        int idx_output_x, idx_output_y, idx_output_z;
        idx_output_x = idx_output % d_outputDim[0];
        idx_output = (idx_output - idx_output_x)/d_outputDim[0];
        idx_output_y = idx_output % d_outputDim[1];
        idx_output_z = (idx_output - idx_output_y)/d_outputDim[1];
        
        int offsetInput = idx_out_feature*d_inputDim[0]*d_inputDim[1]*d_inputDim[2];
        // (x, y, z):       2*idx_output_z*d_inputDim[0]*d_inputDim[1] +        2*idx_output_y*d_inputDim[0] +          2*idx_output_x +        offsetInput;
        // (x+1, y, z):     2*idx_output_z*d_inputDim[0]*d_inputDim[1] +        2*idx_output_y*d_inputDim[0] +          2*idx_output_x + 1 +    offsetInput;
        // (x, y+1, z):     2*idx_output_z*d_inputDim[0]*d_inputDim[1] +        (2*idx_output_y + 1)*d_inputDim[0] +    2*idx_output_x +        offsetInput;
        // (x+1, y+1, z):   2*idx_output_z*d_inputDim[0]*d_inputDim[1] +        (2*idx_output_y + 1)*d_inputDim[0] +    2*idx_output_x + 1 +    offsetInput;
        // (x, y, z+1):     (2*idx_output_z + 1)*d_inputDim[0]*d_inputDim[1] +  2*idx_output_y*d_inputDim[0] +          2*idx_output_x +        offsetInput;
        // (x+1, y, z+1):   (2*idx_output_z + 1)*d_inputDim[0]*d_inputDim[1] +  2*idx_output_y*d_inputDim[0] +          2*idx_output_x + 1 +    offsetInput;
        // (x, y+1, z+1):   (2*idx_output_z + 1)*d_inputDim[0]*d_inputDim[1] +  (2*idx_output_y + 1)*d_inputDim[0] +    2*idx_output_x +        offsetInput;
        // (x+1, y+1, z+1): (2*idx_output_z + 1)*d_inputDim[0]*d_inputDim[1] +  (2*idx_output_y + 1)*d_inputDim[0] +    2*idx_output_x + 1 +    offsetInput;
        
        
        
        int idx_input = 2*(idx_output_z*d_inputDim[0]*d_inputDim[1] + idx_output_y*d_inputDim[0] + idx_output_x) + offsetInput;
        if(d_input[idx_input] > d_input[idx_input + 1]){
            d_output[globalIdxOut] = d_input[idx_input];//(x, y, z)
        }else{
            d_output[globalIdxOut] = d_input[idx_input + 1];//(x+1, y, z)
        }
        
        idx_input += d_inputDim[0];//(x, y+1, z)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
        
        idx_input += 1;//(x+1, y+1, z)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
        
        idx_input += d_inputDim[0]*d_inputDim[1];//(x+1, y+1, z+1)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
        
        idx_input -= d_inputDim[0];//(x+1, y, z+1)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
        
        idx_input -= 1;//(x, y, z+1)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
        
        idx_input += d_inputDim[0];//(x, y+1, z+1)
        if(d_output[globalIdxOut] < d_input[idx_input]){
            d_output[globalIdxOut] = d_input[idx_input];
        }
    }
}


//dedx should be assigned all zero before using this function.
__global__ void GPUMaxPoolingdx2x2x2v2(int* d_inputDim, float* d_input, int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy){
    int idx_dedy = blockIdx.x*blockDim.x + threadIdx.x;
    int idx_dedy_feature = blockIdx.y*blockDim.y + threadIdx.y;
    if((idx_dedy < d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2]) && (idx_dedy_feature < d_dedyDim[3])){
    
        int idx_dedy_x, idx_dedy_y, idx_dedy_z;
        idx_dedy_x = idx_dedy % d_dedyDim[0];
        idx_dedy = (idx_dedy - idx_dedy_x)/d_dedyDim[0];
        idx_dedy_y = idx_dedy % d_dedyDim[1];
        idx_dedy_z = (idx_dedy - idx_dedy_y)/d_dedyDim[1];
        
        int offsetInput = idx_dedy_feature*d_inputDim[0]*d_inputDim[1]*d_inputDim[2];
        // (x, y, z):       2*idx_dedy_z*d_inputDim[0]*d_inputDim[1] +        2*idx_dedy_y*d_inputDim[0] +          2*idx_dedy_x +        offsetInput;
        // (x+1, y, z):     2*idx_dedy_z*d_inputDim[0]*d_inputDim[1] +        2*idx_dedy_y*d_inputDim[0] +          2*idx_dedy_x + 1 +    offsetInput;
        // (x, y+1, z):     2*idx_dedy_z*d_inputDim[0]*d_inputDim[1] +        (2*idx_dedy_y + 1)*d_inputDim[0] +    2*idx_dedy_x +        offsetInput;
        // (x+1, y+1, z):   2*idx_dedy_z*d_inputDim[0]*d_inputDim[1] +        (2*idx_dedy_y + 1)*d_inputDim[0] +    2*idx_dedy_x + 1 +    offsetInput;
        // (x, y, z+1):     (2*idx_dedy_z + 1)*d_inputDim[0]*d_inputDim[1] +  2*idx_dedy_y*d_inputDim[0] +          2*idx_dedy_x +        offsetInput;
        // (x+1, y, z+1):   (2*idx_dedy_z + 1)*d_inputDim[0]*d_inputDim[1] +  2*idx_dedy_y*d_inputDim[0] +          2*idx_dedy_x + 1 +    offsetInput;
        // (x, y+1, z+1):   (2*idx_dedy_z + 1)*d_inputDim[0]*d_inputDim[1] +  (2*idx_dedy_y + 1)*d_inputDim[0] +    2*idx_dedy_x +        offsetInput;
        // (x+1, y+1, z+1): (2*idx_dedy_z + 1)*d_inputDim[0]*d_inputDim[1] +  (2*idx_dedy_y + 1)*d_inputDim[0] +    2*idx_dedy_x + 1 +    offsetInput;
        
        
        
        int idx_dedx = 2*(idx_dedy_z*d_dedxDim[0]*d_dedxDim[1] + idx_dedy_y*d_dedxDim[0] + idx_dedy_x) + offsetInput;
        int tempIdx;
        float tempVal;
        
        if(d_input[idx_dedx] > d_input[idx_dedx + 1]){
            tempIdx = idx_dedx; //(x, y, z)
            tempVal = d_input[idx_dedx];
        }else{
            tempIdx = idx_dedx + 1; //(x+1, y, z)
            tempVal = d_input[idx_dedx + 1];
        }
        
        idx_dedx += d_inputDim[0]; //(x, y+1, z)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        idx_dedx += 1; //(x+1, y+1, z)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        idx_dedx += d_inputDim[0]*d_inputDim[1]; //(x+1, y+1, z+1)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        idx_dedx -= d_inputDim[0]; //(x+1, y, z+1)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        idx_dedx -= 1; //(x, y, z+1)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        idx_dedx += d_inputDim[0]; //(x, y+1, z+1)
        if(tempVal < d_input[idx_dedx]){
            tempIdx = idx_dedx;
            tempVal = d_input[idx_dedx];
        }
        
        d_dedx[tempIdx] = d_dedy[idx_dedy_feature*d_dedyDim[0]*d_dedyDim[1]*d_dedyDim[2] + idx_dedy];
    }
}



// Not tested yet, but had been checked by eyes.
// Here ThR, ThC, and ThD are the thread numbers with respect to d_out_y.
// Consider to include the computation of derivative here to avoid redundant comparison of d_in_x.
__global__ void GPUMaxPooling2x2x2(float* d_in_x, float* d_out_y, int ThR, int ThC, int ThD){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;
    if((row < ThC)&&(col < ThR)&&(layer < ThD)){
        int index = 2*(row + col*2*ThC + layer*4*ThC*ThR);
        float temp;
        if (d_in_x[index] >= d_in_x[index+1]){
            temp = d_in_x[index];
        }else{
            temp = d_in_x[index+1];
        }
        
        index += 2*ThC;
        
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        
        index += 1;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        
        index = 2*(row + col*2*ThC + layer*4*ThC*ThR) + 4*ThC*ThR;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        index += 1;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        index += 2*ThC -1;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        index += 1;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        d_out_y[row + col*ThC + layer*ThR*ThC] = temp;
    }
}

// Not tested yet, but had been checked by eyes.
// ThR and ThC are the thread numbers of output per row and per column, respectively.
__global__ void GPUMaxPooling2x2(float* d_in_x, float* d_out_y, int ThR, int ThC){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if((row < ThC)&&(col < ThR)){
        int index = 2*(row + col*2*ThC);
        float temp;
        if (d_in_x[index] >= d_in_x[index+1]){
            temp = d_in_x[index];
        }else{
            temp = d_in_x[index+1];
        }
        
        index += 2*ThC;
        
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        
        index += 1;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
        }
        d_out_y[row + col*ThC] = temp;
    }
}

// Not tested yet, but had been checked by eyes.
// d_in_x is the input data tensor of the corresponding forward max pooling layer.
// d_out_dedx should be initialized to all zero.
__global__ void GPUMaxPool2x2x2dx(float* d_in_x, float* d_in_dedy, float* d_out_dedx, int ThR, int ThC, int ThD){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;
    if((row < ThC)&&(col < ThR)&&(layer < ThD)){
        int index = 2*(row + col*2*ThC + layer*4*ThC*ThR);
        int maxIndex;
        float temp;
        
        d_out_dedx[index] = 0;
        d_out_dedx[index+1] = 0;
        
        if (d_in_x[index] >= d_in_x[index+1]){
            temp = d_in_x[index];
            maxIndex = index;
        }else{
            temp = d_in_x[index+1];
            maxIndex = index+1;
        }
        index += 2*ThC;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        index += 1;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        
        index = 2*(row + col*2*ThC + layer*4*ThC*ThR) + 4*ThC*ThR;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        index += 1;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        index += 2*ThC -1;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        index += 1;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        
        d_out_dedx[maxIndex] = d_in_dedy[row + col*ThC + layer*ThC*ThR];
    }
}

// Not tested yet, but had been checked by eyes.
// d_in_x is the input data tensor of the corresponding forward max pooling layer.
// d_out_dedx should be initialized to all zero.
__global__ void GPUMaxPool2x2dx(float* d_in_x, float* d_in_dedy, float* d_out_dedx, int ThR, int ThC){
    int row = blockIdx.x*blockDim.x + threadIdx.x; // (row, col) is the coordinate of d_in_dedy.
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    if((row < ThC)&&(col < ThR)){
        int index = 2*(row + 2*col*ThC);
        int maxIndex;
        float temp;
        
        d_out_dedx[index] = 0;
        d_out_dedx[index+1] = 0;
        
        if (d_in_x[index] >= d_in_x[index+1]){
            temp = d_in_x[index];
            maxIndex = index;
        }else{
            temp = d_in_x[index+1];
            maxIndex = index+1;
        }
        
        index += 2*ThC;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        
        index += 1;
        d_out_dedx[index] = 0;
        if (temp < d_in_x[index]){
            temp = d_in_x[index];
            maxIndex = index;
        }
        
        d_out_dedx[maxIndex] = d_in_dedy[row + col*ThC];
    }
}



__global__ void GPUMProd(float* d_inA, float* d_inB, float* d_outC, int dimension1, int dimension2, int dimension3){
    int row = blockIdx.x*blockDim.x + threadIdx.x;//0 1
    int col = blockIdx.y*blockDim.y + threadIdx.y;//0 1 2 3
    float ans = 0;
    if((row < dimension1)&&(col < dimension3)){
        for(int i=0; i<dimension2; i++){//i=0 1 2
            ans += d_inA[row + i*dimension1]*d_inB[col*dimension2 + i];
        }
        d_outC[row + col*dimension1] = ans;
    }
}

// dimension3 is useless. It is just a dreg from GPUMProd();
__global__ void GPUMProdDropOut(float* d_inA, float* d_inB, float* d_outC, int dimension1, int dimension2, int dimension3, float* d_dropIdx){
    int row = blockIdx.x*blockDim.x + threadIdx.x;//0 1
    float ans = 0;
    if(row < dimension1){
        if(d_dropIdx[row] > 0){
            for(int i=0; i<dimension2; i++){//i=0 1 2
                ans += d_inA[row + i*dimension1]*d_inB[i];
            }
            d_outC[row] = ans;
        }else{
            d_outC[row] = 0.0;
        }
    }
}

// dimension1 == 1 is useless. It is just a dreg from GPUMProd();
__global__ void GPUMDropOutBack(float* d_inA, float* d_inB, float* d_outC, int dimension1, int dimension2, int dimension3, float* d_dropIdx){
    //int row = blockIdx.x*blockDim.x + threadIdx.x;//0 1
    int col = blockIdx.y*blockDim.y + threadIdx.y;//0 1 2 3
    float ans = 0;
    if(col < dimension3){
        int off = col*dimension2;
        for(int i=0; i<dimension2; i++){//i=0 1 2
            if(d_dropIdx[i] > 0){
                ans += d_inA[i]*d_inB[off + i];
            }
        }
        d_outC[col] = ans;
    }
}

// For the case of full-connected network,
// this function can be used to compute for w - alpha*de/dw 
// with d_in_dydw=dy/dw=x (the previous input data), d_in_dedy=de/dy.
__global__ void GPUFCdw(float* d_in_dydw, float* d_in_dedy, float* d_FCweight, int dimension1, int dimension2, int dimension3,
                        float alpha){
    int row = blockIdx.x*blockDim.x + threadIdx.x;//0 1
    int col = blockIdx.y*blockDim.y + threadIdx.y;//0 1 2 3
    float ans = 0;
    if((row < dimension1)&&(col < dimension3)){
        for(int i=0; i<dimension2; i++){//i=0 1 2
            ans += d_in_dydw[row + i*dimension1]*d_in_dedy[col*dimension2 + i];
        }
        d_FCweight[col + row*dimension3] -= ans*alpha;
    }
}

__global__ void GPUFCcomputededw(float* d_in_dydw, float* d_in_dedy, float* d_dedw, int dimension1, int dimension3){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    float ans = 0;
    if((row < dimension1)&&(col < dimension3)){
        ans += d_in_dydw[row]*d_in_dedy[col];
        d_dedw[col + row*dimension3] = ans;// Here is a transpose, so the roles played by col and row are switched.
    }
}

__global__ void GPUFCcomputededwDropOut(float* d_in_dydw, float* d_in_dedy, float* d_dedw, int dimension1, int dimension3, float* d_dropIdx){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    float ans = 0;
    if((row < dimension1)&&(col < dimension3)){
        if(d_dropIdx[col] > 0){
            ans += d_in_dydw[row]*d_in_dedy[col];
        }
        d_dedw[col + row*dimension3] = ans;// Here is a transpose, so the roles played by col and row are switched.
    }
}


__global__ void GPUFCAdjust(float* d_FCweight, float* d_dedw, int dimension1, int dimension3, float alpha){
    int row = blockIdx.x*blockDim.x + threadIdx.x;//0 1
    int col = blockIdx.y*blockDim.y + threadIdx.y;//0 1 2 3
    if((row < dimension1)&&(col < dimension3)){
        d_FCweight[col + row*dimension3] -= d_dedw[col + row*dimension3]*alpha;
    }
}

// GPUVectorSum computes sum_{i=1}^vectorN d_dataHeads[i].
__global__ void GPUVectorSum(float** d_dataHeads, float* d_out, int vectorN, int dataL){
    int dataIdx = threadIdx.x + blockIdx.x*blockDim.x;
    float ans = 0;
    if(dataIdx < dataL){
        for(int i=0; i<vectorN; i++){
            ans += d_dataHeads[i][dataIdx];
        }
        d_out[dataIdx] = ans;
    }
}

// d_out should be initialized.
__global__ void GPUVectorAddUp(float* d_in, float* d_out, int dataL){
    int dataIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if(dataIdx < dataL){
        d_out[dataIdx] += d_in[dataIdx];
    }
}

__global__ void GPUVectorScalarAddUp(float* d_in, float* d_out, int dataL, float alpha){
    int dataIdx = threadIdx.x + blockIdx.x*blockDim.x;
    if(dataIdx < dataL){
        d_out[dataIdx] += alpha*d_in[dataIdx];
    }
}


// Not tested yet.
__global__ void GPUBias(float* d_in_x, float* d_out_y, float* bias, int ThR){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < ThR){
        d_out_y[index] = d_in_x[index] + bias[index];
    }
}

__global__ void GPUBiasDropOut(float* d_in_x, float* d_out_y, float* bias, int ThR, float* d_dropIdx){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < ThR){
        if (d_dropIdx[index] > 0){
            d_out_y[index] = d_in_x[index] + bias[index];
        }else{
            d_out_y[index] = 0.0;
        }
    }
}

__global__ void GPUBiasdedw(float* d_dedw, float* d_dedy, float* d_dropIdx, int ThR){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < ThR){
        if (d_dropIdx[index] > 0){
            d_dedw[index] = d_dedy[index];
        }else{
            d_dedw[index] = 0.0;
        }
    }
}

// de/dw = de/dy, because dy/dw is just the identity matrix.
__global__ void GPUBiasAdjust(float* d_in_dedw, float* bias, int ThR, float alpha){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < ThR){
        bias[index] -= d_in_dedw[index]*alpha;
    }
}

// Not tested yet.
__global__ void GPUReLU(float* d_in_x, float* d_out_y, int ThR){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < ThR){
        if (d_in_x[index] > 0){
            d_out_y[index] = d_in_x[index];
        }else{
            d_out_y[index] = 0;
        }
    }
}

// Not tested yet. d_in_x is the output data tensor of this corresponding forward ReLU layer.
__global__ void GPUReLUdx(float* d_in_dedy, float* d_in_x, float* d_out_dedx, int ThR){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if (index < ThR){
        if (d_in_x[index] <= 0.00001){
            d_out_dedx[index] = 0;
        }else{
            d_out_dedx[index] = d_in_dedy[index];
        }
    }
}

//====================================================================================
			// no use //

__global__ void vadd(float * d_out, float * d_in1, float * d_in2){
    int idx = threadIdx.x;
    float a = d_in1[idx];
    float b = d_in2[idx];
    d_out[idx] = a + b;
}

__global__ void GPUconv1D(float* d_inData, float* d_inFilter, float* d_outData, int xLength, int filterXLength, int xPadding){
    int idx = threadIdx.x;
    float ans = 0;
    int currentIdx = idx - xPadding;
    for(int i=0; i<filterXLength; i++){
        if((currentIdx >= 0)&&(currentIdx < xLength)){
            ans += d_inFilter[i]*d_inData[currentIdx];
        }
        currentIdx++;
    }
    d_outData[idx] = ans;
    /*if (idx-xPaddingN < 0){
        
    }else if(idx+xPaddingN > xLength-1){

    }else{

    }*/
}

// GPUMProdT is the same with GPUProd with applying a transpose to the output.
// For the case of full-connected network,
// this function can be used to compute for de/dw 
// with d_inA=df/dw=x (the previous input data), d_inB=de/df, and d_outC=de/dw.
// Use GPUFCdw instead.
__global__ void GPUMProdT(float* d_inA, float* d_inB, float* d_outC, int BLx, int BLy, int dimension1, int dimension2, int dimension3){
    int row = blockIdx.x*BLx + threadIdx.x;//0 1
    int col = blockIdx.y*BLy + threadIdx.y;//0 1 2 3
    float ans = 0;
    if((row < dimension1)&&(col < dimension3)){
        for(int i=0; i<dimension2; i++){//i=0 1 2
            ans += d_inA[row + i*dimension1]*d_inB[col*dimension2 + i];
        }
        d_outC[col + row*dimension3] = ans;
    }
}

// Not tested yet. This function should be replaced by GPUReLU.
__global__ void GPUReLUConv(float* d_inData, float* d_outData, int ThR, int ThC, int ThD){
    int row = blockIdx.x*blockDim.x + threadIdx.x;
    int col = blockIdx.y*blockDim.y + threadIdx.y;
    int layer = blockIdx.z*blockDim.z + threadIdx.z;
    if((row < ThR)&&(col < ThC)&&(layer < ThD)){
        int index = row + col*ThR + layer*ThR*ThC;
        if(d_inData[index]>0){
            d_outData[index] = d_inData[index];
        }else{
            d_outData[index] = 0;
        }
    }
}

//====================================================================================
// class


    void ProtoLayer::setAll(){}
    void ProtoLayer::forward(){
        cout << "Dummy forwarding.\n";
    }
    void ProtoLayer::backward(){}
    float ProtoLayer::forward(int a){cout << "Dummy forwarding.\n"; return 0;}
    float ProtoLayer::forward(float* a){cout << "Dummy forwarding.\n"; return 0;}
    void ProtoLayer::backward(int a){}
    
    void ProtoLayer::adjust(float alpha, float batchSize){
        cout << "This layer does not adjust weights.\n";
    }
    
    void ProtoLayer::setInput(dataTensor* in_x){
        input = in_x;
    }
    void ProtoLayer::setOutput(dataTensor* in_y){
        output = in_y;
    }
    void ProtoLayer::setdedx(dataTensor* in_dedx){
        dedx = in_dedx;
    }
    void ProtoLayer::setdedy(dataTensor* in_dedy){
        dedy = in_dedy;
    }
    void ProtoLayer::setWeight(dataTensor* in_weight){
        if (layerType %7 == 6 ){
            weight = in_weight;
        }else{
            cout << "This layer does not need weights.\n";
        }
    }
    
void ProtoLayer::setDropOutRatio(float in_ratio){
    dropOutRatio = in_ratio;
}
    
dataTensor* ProtoLayer::getWeight(){
    return(weight);
}

void ProtoLayer::loadWeight(int dataLength, float* in_weight){
    if(this->weight->dataLength() == dataLength){
        weight->resetData(in_weight);
    }else{
        cout << "Weight length not consistent.\n";
    }
}


ProtoLayer::ProtoLayer(){
    layerType = PROTO;
    input = NULL;
    output = NULL;
    dedx = NULL;
    dedy = NULL;
    dedw = NULL;
    alldedw = NULL;
    weight = NULL;
    filterDim = NULL;
    inputDim = NULL;
    outputDim = NULL;
}

void ProtoLayer::setInputDim(int dimN, int* dim){
    inputDimN = dimN;
    if(inputDim != NULL){
        delete[] inputDim;
    }
    inputDim = new int[dimN];
    for(int i=0; i<dimN; i++){
        inputDim[i] = dim[i];
    }
}

void ProtoLayer::setOutputDim(int dimN, int* dim){
    outputDimN = dimN;
    if(inputDim != NULL){
        delete[] outputDim;
    }
    outputDim = new int[dimN];
    for(int i=0; i<dimN; i++){
        outputDim[i] = dim[i];
    }
}

void ProtoLayer::checkAndCorrectInputDim(){
    //cout << "checkAndCorrectInputDim()\n";
    if(!(input->checkDimEqual(inputDimN, inputDim))){
        input->setDim(inputDimN, inputDim);
    }
}

void ProtoLayer::checkAndCorrectOutputDim(){
    //cout << "checkAndCorrectOutputDim()\n";
    if(!(output->checkDimEqual(outputDimN, outputDim))){
        output->setDim(outputDimN, outputDim);
    }
}

void ProtoLayer::checkAndCorrectdedxDim(){
    //cout << "checkAndCorrectdedxDim()\n";
    if(!(dedx->checkDimEqual(inputDimN, inputDim))){
        dedx->setDim(inputDimN, inputDim);
    }
}

void ProtoLayer::checkAndCorrectdedyDim(){
    //cout << "checkAndCorrectdedyDim()\n";
    if(!(dedy->checkDimEqual(outputDimN, outputDim))){
        dedy->setDim(outputDimN, outputDim);
    }
}

dataTensor* ProtoLayer::get_dedw(){return(NULL);}

int ProtoLayer::getType(){
    return(layerType);
}

void ProtoLayer::add_input(dataTensor* in_x){}
void ProtoLayer::add_dedx(dataTensor* dedx){}
void ProtoLayer::add_output(dataTensor* out_y){}
void ProtoLayer::add_dedy(dataTensor* dedy){}

//=========================================================================================

Conv1DLayer::Conv1DLayer(){
    layerType = CONV1D;
}

void Conv1DLayer::setWeight(int xdim){
    filterDim = new int[3];
    filterDim[0] = xdim;
}

void Conv1DLayer::setAll(){
    int tempPadding[1];
    tempPadding[0] = (filterDim[0] - 1)/2;
    
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    
    
    cudaError_t errmsg = cudaMalloc((void **) &d_paddingN, sizeof(int));
    if(errmsg == cudaErrorMemoryAllocation){
        cout << "SeparateOnetoNLayer::setAll() cudaMalloc Error!\n";
        ALLBREAK = 1;
    }
    cudaMemcpy(d_paddingN, tempPadding, sizeof(int), cudaMemcpyHostToDevice);
    
    
    in_featureN = input->dimension[1];
    out_featureN = output->dimension[1];
    
    
    filterDim[1] = in_featureN;
    filterDim[2] = out_featureN;
    
    
    weight = new dataTensor(3, filterDim);
    
    
    BNxf = ceil((float) output->dimension[0]/32.0); // length of output data of a single feature map
    BNyf = ceil((float) out_featureN/32.0); // feature map number of output
    
    BNxb = ceil((float) dedx->dimension[0]/32.0); // length of dedx data of a single feature map
    BNyb = ceil((float) in_featureN/32.0); // feature map number of dedx
    
    BNx = ceil((float) weight->dimension[0]/8.0); // length of weight of a single feature map
    BNy = ceil((float) in_featureN/16.0); // feature map number of input
    BNz = ceil((float) out_featureN/8.0); // feature map number of output
    
    BN = ceil((float) weight->dataLength()/1024.0);
    
    dedw = new dataTensor(weight->dimension[0], in_featureN, out_featureN);
    alldedw = new dataTensor(weight->dimension[0], in_featureN, out_featureN);
    alldedw->zeroAssign();
    
}
    
void Conv1DLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUConv1Dv2(int* d_inputDim, float* d_input, int* d_weightDim, float* d_weight, int* d_outputDim, float* d_output, int paddingN)
    GPUConv1Dv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, weight->d_dimension, weight->d_data, output->d_dimension, output->d_data, d_paddingN);
}

void Conv1DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    //GPUConv1Ddxv2(int* d_dedxDim, float* d_dedx, int* d_weightDim, float* d_weight, int* d_dedyDim, float* d_dedy, int* paddingN)
    GPUConv1Ddxv2<<<dim3(BNxb, BNyb, 1), dim3(32, 32, 1)>>>(dedx->d_dimension, dedx->d_data, weight->d_dimension, weight->d_data, dedy->d_dimension, dedy->d_data, d_paddingN);
    tensorSum(alldedw, get_dedw(), 1);
}
    
dataTensor* Conv1DLayer::get_dedw(){
    // compute dedw
    //GPUConv1Ddwv2(int* d_inputDim, float* d_input, int* d_dedwDim, float* d_dedw, int* d_dedyDim, float* d_dedy, int* paddingN)
    GPUConv1Ddwv2<<<dim3(BNx, BNy, BNz), dim3(8, 16, 8)>>>(input->d_dimension, input->d_data, dedw->d_dimension, dedw->d_data, dedy->d_dimension, dedy->d_data, d_paddingN);
    return(dedw);
}
    
    
void Conv1DLayer::adjust(float alpha, float batchSize){
    GPUVectorScalarAddUp<<<BN, 1024>>>(alldedw->d_data, weight->d_data, weight->dataLength(), alpha*(-1)/batchSize);
    alldedw->zeroAssign();
}
    
//========================================================================================================================



//=========================================================================================

Conv2DLayer::Conv2DLayer(){
    layerType = CONV2D;
}

void Conv2DLayer::setWeight(int xdim, int ydim){
    filterDim = new int[4];
    filterDim[0] = xdim;
    filterDim[1] = ydim;
}

void Conv2DLayer::setAll(){
    int tempPadding[2];
    tempPadding[0] = (filterDim[0] - 1)/2;
    tempPadding[1] = (filterDim[1] - 1)/2;
    
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    
    cudaError_t errmsg = cudaMalloc((void **) &d_paddingN, 2*sizeof(int));
    if(errmsg == cudaErrorMemoryAllocation){
        cout << "SeparateOnetoNLayer::setAll() cudaMalloc Error!\n";
        ALLBREAK = 1;
    }
    cudaMemcpy(d_paddingN, tempPadding, 2*sizeof(int), cudaMemcpyHostToDevice);
    
    
    in_featureN = input->dimension[2];
    out_featureN = output->dimension[2];
    
    
    filterDim[2] = in_featureN;
    filterDim[3] = out_featureN;
    
    
    weight = new dataTensor(4, filterDim);
    
    
    BNxf = ceil((float) output->dimension[0]*output->dimension[1]/32.0); // length of output data of a single feature map
    BNyf = ceil((float) out_featureN/32.0); // feature map number of output
    
    BNxb = ceil((float) dedx->dimension[0]*dedx->dimension[1]/32.0); // length of dedx data of a single feature map
    BNyb = ceil((float) in_featureN/32.0); // feature map number of dedx
    
    BNx = ceil((float) weight->dimension[0]*weight->dimension[1]/8.0); // length of weight of a single feature map
    BNy = ceil((float) in_featureN/16.0); // feature map number of input
    BNz = ceil((float) out_featureN/8.0); // feature map number of output
    
    BN = ceil((float) weight->dataLength()/1024.0);
    
    dedw = new dataTensor(weight->dimension[0], weight->dimension[1], in_featureN, out_featureN);
    alldedw = new dataTensor(weight->dimension[0], weight->dimension[1], in_featureN, out_featureN);
    alldedw->zeroAssign();
    
}
    
void Conv2DLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUConv2Dv2(int* d_inputDim, float* d_input, int* d_weightDim, float* d_weight, int* d_outputDim, float* d_output, int paddingN)
    GPUConv2Dv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, weight->d_dimension, weight->d_data, output->d_dimension, output->d_data, d_paddingN);
}

void Conv2DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    //GPUConv2Ddxv2(int* d_dedxDim, float* d_dedx, int* d_weightDim, float* d_weight, int* d_dedyDim, float* d_dedy, int* paddingN)
    GPUConv2Ddxv2<<<dim3(BNxb, BNyb, 1), dim3(32, 32, 1)>>>(dedx->d_dimension, dedx->d_data, weight->d_dimension, weight->d_data, dedy->d_dimension, dedy->d_data, d_paddingN);
    tensorSum(alldedw, get_dedw(), 1);
}
    
dataTensor* Conv2DLayer::get_dedw(){
    // compute dedw
    //GPUConv2Ddwv2(int* d_inputDim, float* d_input, int* d_dedwDim, float* d_dedw, int* d_dedyDim, float* d_dedy, int* paddingN)
    GPUConv2Ddwv2<<<dim3(BNx, BNy, BNz), dim3(8, 16, 8)>>>(input->d_dimension, input->d_data, dedw->d_dimension, dedw->d_data, dedy->d_dimension, dedy->d_data, d_paddingN);
    return(dedw);
}
    
    
void Conv2DLayer::adjust(float alpha, float batchSize){
    GPUVectorScalarAddUp<<<BN, 1024>>>(alldedw->d_data, weight->d_data, weight->dataLength(), alpha*(-1)/batchSize);
    alldedw->zeroAssign();
}
    
//========================================================================================================================



//=========================================================================================

Conv3DLayer::Conv3DLayer(){
    layerType = CONV3D;
}

void Conv3DLayer::setWeight(int xdim, int ydim, int zdim){
    filterDim = new int[5];
    filterDim[0] = xdim;
    filterDim[1] = ydim;
    filterDim[2] = zdim;
}

void Conv3DLayer::setAll(){
    int tempPadding[3];
    tempPadding[0] = (filterDim[0] - 1)/2;
    tempPadding[1] = (filterDim[1] - 1)/2;
    tempPadding[2] = (filterDim[2] - 1)/2;
    
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    
    cudaError_t errmsg = cudaMalloc((void **) &d_paddingN, 3*sizeof(int));
    if(errmsg == cudaErrorMemoryAllocation){
        cout << "SeparateOnetoNLayer::setAll() cudaMalloc Error!\n";
        ALLBREAK = 1;
    }
    cudaMemcpy(d_paddingN, tempPadding, 3*sizeof(int), cudaMemcpyHostToDevice);
    
    
    in_featureN = input->dimension[3];
    out_featureN = output->dimension[3];
    
    
    filterDim[3] = in_featureN;
    filterDim[4] = out_featureN;
    
    
    weight = new dataTensor(5, filterDim);
    
    
    BNxf = ceil((float) output->dimension[0]*output->dimension[1]*output->dimension[2]/32.0); // length of output data of a single feature map
    BNyf = ceil((float) out_featureN/32.0); // feature map number of output
    
    BNxb = ceil((float) dedx->dimension[0]*dedx->dimension[1]*dedx->dimension[2]/32.0); // length of dedx data of a single feature map
    BNyb = ceil((float) in_featureN/32.0); // feature map number of dedx
    
    BNx = ceil((float) weight->dimension[0]*weight->dimension[1]*weight->dimension[2]/8.0); // length of weight of a single feature map
    BNy = ceil((float) in_featureN/16.0); // feature map number of input
    BNz = ceil((float) out_featureN/8.0); // feature map number of output
    
    BN = ceil((float) weight->dataLength()/1024.0);
    
    dedw = new dataTensor(weight->dimension[0], weight->dimension[1], weight->dimension[2], in_featureN, out_featureN);
    alldedw = new dataTensor(weight->dimension[0], weight->dimension[1], weight->dimension[2], in_featureN, out_featureN);
    alldedw->zeroAssign();
    
}
    
void Conv3DLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUConv3Dv2(int* d_inputDim, float* d_input, int* d_weightDim, float* d_weight, int* d_outputDim, float* d_output, int paddingN)
    GPUConv3Dv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, weight->d_dimension, weight->d_data, output->d_dimension, output->d_data, d_paddingN);
}

void Conv3DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    //GPUConv3Ddxv2(int* d_dedxDim, float* d_dedx, int* d_weightDim, float* d_weight, int* d_dedyDim, float* d_dedy, int* paddingN)
    GPUConv3Ddxv2<<<dim3(BNxb, BNyb, 1), dim3(32, 32, 1)>>>(dedx->d_dimension, dedx->d_data, weight->d_dimension, weight->d_data, dedy->d_dimension, dedy->d_data, d_paddingN);
    tensorSum(alldedw, get_dedw(), 1);
}
    
dataTensor* Conv3DLayer::get_dedw(){
    // compute dedw
    //GPUConv3Ddwv2(int* d_inputDim, float* d_input, int* d_dedwDim, float* d_dedw, int* d_dedyDim, float* d_dedy, int* paddingN)
    GPUConv3Ddwv2<<<dim3(BNx, BNy, BNz), dim3(8, 16, 8)>>>(input->d_dimension, input->d_data, dedw->d_dimension, dedw->d_data, dedy->d_dimension, dedy->d_data, d_paddingN);
    return(dedw);
}
    
    
void Conv3DLayer::adjust(float alpha, float batchSize){
    GPUVectorScalarAddUp<<<BN, 1024>>>(alldedw->d_data, weight->d_data, weight->dataLength(), alpha*(-1)/batchSize);
    alldedw->zeroAssign();
}
    
//========================================================================================================================

Conv1DBiasLayer::Conv1DBiasLayer(){
    layerType = CONV1DBIAS;
}

void Conv1DBiasLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    
    tempLength = input->dimension[0];
    featureN = input->dimension[1];
    weight = new dataTensor(featureN);
    dedw = new dataTensor(featureN);
    alldedw = new dataTensor(featureN);
    alldedw->zeroAssign();
    
        
    BNxf = ceil((float) tempLength/32.0);
    BNyf = ceil((float) featureN/32.0);
        
}
    
    
void Conv1DBiasLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUBiasConv1Dv2(int* d_inputDim, float* d_input, float* d_bias, int* d_outputDim, float* d_output)
    GPUBiasConv1Dv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, weight->d_data, output->d_dimension, output->d_data);
}


void Conv1DBiasLayer::backward(){
    tensorSum(alldedw, get_dedw(), 1);
}
    
dataTensor* Conv1DBiasLayer::get_dedw(){
    dedw->zeroAssign();
    //GPUConvBiasdedwv2(float* d_dedw, float* d_dedy, int dataLength, int featureN)
    GPUConvBiasdedwv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(dedw->d_data, dedy->d_data, tempLength, featureN);
    return(dedw);
}
    
    
    
void Conv1DBiasLayer::adjust(float alpha, float batchSize){
    GPUVectorScalarAddUp<<<1, featureN>>>(alldedw->d_data, weight->d_data, featureN, alpha*(-1)/batchSize);
    alldedw->zeroAssign();
}



//========================================================================================================================


//========================================================================================================================

Conv2DBiasLayer::Conv2DBiasLayer(){
    layerType = CONV2DBIAS;
}

void Conv2DBiasLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    tempLength = input->dimension[0]*input->dimension[1];
    featureN = input->dimension[2];
    weight = new dataTensor(featureN);
    dedw = new dataTensor(featureN);
    alldedw = new dataTensor(featureN);
    alldedw->zeroAssign();
    
        
    BNxf = ceil((float) tempLength/32.0);
    BNyf = ceil((float) featureN/32.0);
        
}
    
    
void Conv2DBiasLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUBiasConv2Dv2(int* d_inputDim, float* d_input, float* d_bias, int* d_outputDim, float* d_output)
    GPUBiasConv2Dv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, weight->d_data, output->d_dimension, output->d_data);
}


void Conv2DBiasLayer::backward(){
    tensorSum(alldedw, get_dedw(), 1);
}
    
dataTensor* Conv2DBiasLayer::get_dedw(){
    dedw->zeroAssign();
    //GPUConvBiasdedwv2(float* d_dedw, float* d_dedy, int dataLength, int featureN)
    GPUConvBiasdedwv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(dedw->d_data, dedy->d_data, tempLength, featureN);
    return(dedw);
}
    
    
    
void Conv2DBiasLayer::adjust(float alpha, float batchSize){
    GPUVectorScalarAddUp<<<1, featureN>>>(alldedw->d_data, weight->d_data, featureN, alpha*(-1)/batchSize);
    alldedw->zeroAssign();
}



//========================================================================================================================
    



//========================================================================================================================

Conv3DBiasLayer::Conv3DBiasLayer(){
    layerType = CONV3DBIAS;
}

void Conv3DBiasLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    tempLength = input->dimension[0]*input->dimension[1]*input->dimension[2];
    featureN = input->dimension[3];
    weight = new dataTensor(featureN);
    dedw = new dataTensor(featureN);
    alldedw = new dataTensor(featureN);
    alldedw->zeroAssign();
    
        
    BNxf = ceil((float) tempLength/32.0);
    BNyf = ceil((float) featureN/32.0);
        
}
    
    
void Conv3DBiasLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUBiasConv3Dv2(int* d_inputDim, float* d_input, float* d_bias, int* d_outputDim, float* d_output)
    GPUBiasConv3Dv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, weight->d_data, output->d_dimension, output->d_data);
}


void Conv3DBiasLayer::backward(){
    tensorSum(alldedw, get_dedw(), 1);
}
    
dataTensor* Conv3DBiasLayer::get_dedw(){
    dedw->zeroAssign();
    //GPUConvBiasdedwv2(float* d_dedw, float* d_dedy, int dataLength, int featureN)
    GPUConvBiasdedwv2<<<dim3(BNxf, BNyf, 1), dim3(32, 32, 1)>>>(dedw->d_data, dedy->d_data, tempLength, featureN);
    return(dedw);
}
    
    
    
void Conv3DBiasLayer::adjust(float alpha, float batchSize){
    GPUVectorScalarAddUp<<<1, featureN>>>(alldedw->d_data, weight->d_data, featureN, alpha*(-1)/batchSize);
    alldedw->zeroAssign();
}



//========================================================================================================================
    
    
//====================================================================================

CBR1DLayer::CBR1DLayer(){
    layerType = CBR1D;
    conv = NULL;
    bias = NULL;
    relu = NULL;
    Conv2Bias = NULL;
    Bias2ReLU = NULL;
    Bias2Conv = NULL; // Bias1Conv is unecessary, because dedx = dedy for bias.
    ReLU2Bias = NULL;
    ConvWeight = NULL;
    BiasWeight = NULL;
    allWeight = NULL;
}

void CBR1DLayer::setWeight(int xdim){
    filterDim = new int[3];
    filterDim[0] = xdim;
}

void CBR1DLayer::setAll(){
    conv = new Conv1DLayer;
    bias = new Conv1DBiasLayer;
    relu = new ReLULayer;
    
    Conv2Bias = new dataTensor(outputDimN, outputDim);
    Bias2ReLU = new dataTensor(outputDimN, outputDim);
    ReLU2Bias = new dataTensor(outputDimN, outputDim);
    //Bias2Conv = new dataTensor(output->dimensionN, output->dimension);
    //ConvWeight = weight;
    
    conv->setInputDim(inputDimN, inputDim);
    conv->setOutputDim(outputDimN, outputDim);
    bias->setInputDim(outputDimN, outputDim);
    bias->setOutputDim(outputDimN, outputDim);
    relu->setInputDim(outputDimN, outputDim);
    relu->setOutputDim(outputDimN, outputDim);
    
    conv->setInput(input);
    conv->setOutput(Conv2Bias);
    conv->setdedx(dedx);
    conv->setdedy(ReLU2Bias);
    conv->setWeight(filterDim[0]);
    conv->setAll();
    
    bias->setInput(Conv2Bias);
    bias->setOutput(Bias2ReLU);
    bias->setdedx(ReLU2Bias);
    bias->setdedy(ReLU2Bias);
    bias->setAll();
    
    
    relu->setInput(Bias2ReLU);
    relu->setOutput(output);
    relu->setdedx(ReLU2Bias);
    relu->setdedy(dedy);
    relu->setAll();
    
    ConvWeight = conv->weight;
    BiasWeight = bias->weight;
}

void CBR1DLayer::forward(){
    conv->forward();
    bias->forward();
    relu->forward();
}

void CBR1DLayer::backward(){
    relu->backward();
    bias->backward();
    conv->backward();
}

void CBR1DLayer::adjust(float alpha, float batchSize){
    conv->adjust(alpha, batchSize);
    bias->adjust(alpha, batchSize);
}

dataTensor* CBR1DLayer::getWeight(){
    int weightLength = ConvWeight->dataLength() + BiasWeight->dataLength();
    if (allWeight == NULL){
        allWeight = new dataTensor(weightLength);
    }
    float* h_weight = new float[weightLength];
    float* h_ConvWeight = ConvWeight->getData();
    float* h_BiasWeight = BiasWeight->getData();
    for(i=0; i<ConvWeight->dataLength(); i++){
        h_weight[i] = h_ConvWeight[i];
    }
    for(i=ConvWeight->dataLength(); i<weightLength; i++){
        h_weight[i] = h_BiasWeight[i - ConvWeight->dataLength()];
    }
    allWeight->resetData(h_weight);
    delete[] h_weight;
    delete[] h_ConvWeight;
    delete[] h_BiasWeight;
    h_weight = NULL;
    return(allWeight);
}

void CBR1DLayer::loadWeight(int dataLength, float* in_weight){
    int weightLength = ConvWeight->dataLength() + BiasWeight->dataLength();
    if(weightLength == dataLength){
        ConvWeight->resetData(in_weight);
        BiasWeight->resetData(in_weight + ConvWeight->dataLength());
    }else{
        cout << "CBR2DLayer::loadWeight() dataLength not consistent.\n";
    }
}    
   
//====================================================================================

//====================================================================================

CBR2DLayer::CBR2DLayer(){
    layerType = CBR2D;
    conv = NULL;
    bias = NULL;
    relu = NULL;
    Conv2Bias = NULL;
    Bias2ReLU = NULL;
    Bias2Conv = NULL; // Bias2Conv is unecessary, because dedx = dedy for bias.
    ReLU2Bias = NULL;
    ConvWeight = NULL;
    BiasWeight = NULL;
    allWeight = NULL;
}

void CBR2DLayer::setWeight(int xdim, int ydim){
    filterDim = new int[4];
    filterDim[0] = xdim;
    filterDim[1] = ydim;
}

void CBR2DLayer::setAll(){
    conv = new Conv2DLayer;
    bias = new Conv2DBiasLayer;
    relu = new ReLULayer;
    
    Conv2Bias = new dataTensor(outputDimN, outputDim);
    Bias2ReLU = new dataTensor(outputDimN, outputDim);
    ReLU2Bias = new dataTensor(outputDimN, outputDim);
    
    
    //Bias2Conv = new dataTensor(output->dimensionN, output->dimension);
    //ConvWeight = weight;
    
    conv->setInputDim(inputDimN, inputDim);
    conv->setOutputDim(outputDimN, outputDim);
    bias->setInputDim(outputDimN, outputDim);
    bias->setOutputDim(outputDimN, outputDim);
    relu->setInputDim(outputDimN, outputDim);
    relu->setOutputDim(outputDimN, outputDim);
    
    conv->setInput(input);
    conv->setOutput(Conv2Bias);
    conv->setdedx(dedx);
    conv->setdedy(ReLU2Bias);
    conv->setWeight(filterDim[0], filterDim[1]);
    conv->setAll();
    
    bias->setInput(Conv2Bias);
    bias->setOutput(Bias2ReLU);
    bias->setdedx(ReLU2Bias);
    bias->setdedy(ReLU2Bias);
    bias->setAll();
    
    
    relu->setInput(Bias2ReLU);
    relu->setOutput(output);
    relu->setdedx(ReLU2Bias);
    relu->setdedy(dedy);
    relu->setAll();
    
    
    
    ConvWeight = conv->weight;
    BiasWeight = bias->weight;
}

void CBR2DLayer::forward(){
    conv->forward();
    bias->forward();
    relu->forward();
    
/*    cout << "CBR2DLayer::forward(): id: " << this->id << endl;
    relu->output->showDim();
    relu->output->showData();*/
}

void CBR2DLayer::backward(){
    relu->backward();
    bias->backward();
    conv->backward();
}

void CBR2DLayer::adjust(float alpha, float batchSize){
    conv->adjust(alpha, batchSize);
    bias->adjust(alpha, batchSize);
}

dataTensor* CBR2DLayer::getWeight(){
    int weightLength = ConvWeight->dataLength() + BiasWeight->dataLength();
    if (allWeight == NULL){
        allWeight = new dataTensor(weightLength);
    }
    float* h_weight = new float[weightLength];
    float* h_ConvWeight = ConvWeight->getData();
    float* h_BiasWeight = BiasWeight->getData();
    for(i=0; i<ConvWeight->dataLength(); i++){
        h_weight[i] = h_ConvWeight[i];
    }
    for(i=ConvWeight->dataLength(); i<weightLength; i++){
        h_weight[i] = h_BiasWeight[i - ConvWeight->dataLength()];
    }
    allWeight->resetData(h_weight);
    delete[] h_weight;
    delete[] h_ConvWeight;
    delete[] h_BiasWeight;
    h_weight = NULL;
    return(allWeight);
}

void CBR2DLayer::loadWeight(int dataLength, float* in_weight){
    int weightLength = ConvWeight->dataLength() + BiasWeight->dataLength();
    if(weightLength == dataLength){
        ConvWeight->resetData(in_weight);
        BiasWeight->resetData(in_weight + ConvWeight->dataLength());
    }else{
        cout << "CBR2DLayer::loadWeight() dataLength not consistent.\n";
    }
}    
   
//====================================================================================


CBR3DLayer::CBR3DLayer(){
    layerType = CBR3D;
    conv = NULL;
    bias = NULL;
    relu = NULL;
    Conv2Bias = NULL;
    Bias2ReLU = NULL;
    Bias2Conv = NULL;
    ReLU2Bias = NULL;
    ConvWeight = NULL;
    BiasWeight = NULL;
    allWeight = NULL;
}

void CBR3DLayer::setWeight(int xdim, int ydim, int zdim){
    filterDim = new int[5];
    filterDim[0] = xdim;
    filterDim[1] = ydim;
    filterDim[2] = zdim;
}

void CBR3DLayer::setAll(){

    conv = new Conv3DLayer;
    bias = new Conv3DBiasLayer;
    relu = new ReLULayer;
    
    Conv2Bias = new dataTensor(outputDimN, outputDim);
    Bias2ReLU = new dataTensor(outputDimN, outputDim);
    ReLU2Bias = new dataTensor(outputDimN, outputDim);
    
    conv->setInputDim(inputDimN, inputDim);
    conv->setOutputDim(outputDimN, outputDim);
    bias->setInputDim(outputDimN, outputDim);
    bias->setOutputDim(outputDimN, outputDim);
    relu->setInputDim(outputDimN, outputDim);
    relu->setOutputDim(outputDimN, outputDim);
    
    conv->setInput(input);
    conv->setOutput(Conv2Bias);
    conv->setdedx(dedx);
    conv->setdedy(ReLU2Bias);
    conv->setWeight(filterDim[0], filterDim[1], filterDim[2]);
    conv->setAll();
    
    
    bias->setInput(Conv2Bias);
    bias->setOutput(Bias2ReLU);
    bias->setdedx(ReLU2Bias);
    bias->setdedy(ReLU2Bias);
    bias->setAll();
    
    
    relu->setInput(Bias2ReLU);
    relu->setOutput(output);
    relu->setdedx(ReLU2Bias);
    relu->setdedy(dedy);
    relu->setAll();
    
    
    
    
    ConvWeight = conv->weight;
    BiasWeight = bias->weight;
    
    
}

void CBR3DLayer::forward(){
    conv->forward();
    bias->forward();
    relu->forward();
}

void CBR3DLayer::backward(){
    relu->backward();
    bias->backward();
    conv->backward();
}

void CBR3DLayer::adjust(float alpha, float batchSize){
    conv->adjust(alpha, batchSize);
    bias->adjust(alpha, batchSize);
}

dataTensor* CBR3DLayer::getWeight(){
    int weightLength = ConvWeight->dataLength() + BiasWeight->dataLength();
    if (allWeight == NULL){
        //cout << "CBR3DLayer::getWeight() Create allWeight.\n";
        allWeight = new dataTensor(weightLength);
    }
    float* h_weight = new float[weightLength];
    float* h_ConvWeight = ConvWeight->getData();
    float* h_BiasWeight = BiasWeight->getData();
    for(i=0; i<ConvWeight->dataLength(); i++){
        h_weight[i] = h_ConvWeight[i];
    }
    for(i=ConvWeight->dataLength(); i<weightLength; i++){
        h_weight[i] = h_BiasWeight[i - ConvWeight->dataLength()];
    }
    allWeight->resetData(h_weight);
    delete[] h_weight;
    delete[] h_ConvWeight;
    delete[] h_BiasWeight;
    h_weight = NULL;
    return(allWeight);
}

void CBR3DLayer::loadWeight(int dataLength, float* in_weight){
    int weightLength = ConvWeight->dataLength() + BiasWeight->dataLength();
    if(weightLength == dataLength){
        ConvWeight->resetData(in_weight);
        BiasWeight->resetData(in_weight + ConvWeight->dataLength());
    }else{
        cout << "CBR3DLayer::loadWeight() dataLength not consistent.\n";
    }
}

//====================================================================================

MaxPooling2Layer::MaxPooling2Layer(){
    layerType = MAXPOOLING2;
}

void MaxPooling2Layer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) output->dimension[0]/32.0);
    BNy = ceil((float) output->dimension[1]/32.0);
    if((input->dimension[0] %2 != 0)){
        cout << "MaxPooling2Layer::setAll(): Max pooling dimension wrong.\n";
    }
}


void MaxPooling2Layer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUMaxPooling2v2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output)
    GPUMaxPooling2v2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, output->d_dimension, output->d_data);
}
    

void MaxPooling2Layer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    dedx->zeroAssign();
    //GPUMaxPoolingdx2v2(int* d_inputDim, float* d_input, int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy)
    GPUMaxPoolingdx2v2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, dedx->d_dimension, dedx->d_data, dedy->d_dimension, dedy->d_data);
}


//====================================================================================

//====================================================================================

MaxPooling2x2Layer::MaxPooling2x2Layer(){
    layerType = MAXPOOLING2X2;
}

void MaxPooling2x2Layer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) output->dimension[0]*output->dimension[1]/32.0);
    BNy = ceil((float) output->dimension[2]/32.0);
    if((input->dimension[0] %2 != 0) || (input->dimension[1] %2 != 0)){
        cout << "MaxPooling2x2Layer::setAll(): Max pooling dimension wrong.\n";
    }
}


void MaxPooling2x2Layer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUMaxPooling2x2v2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output)
    GPUMaxPooling2x2v2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, output->d_dimension, output->d_data);
    
    /*cout << "MaxPooling2x2Layer::forward(): id: " << this->id << " output:\n";
    output->showDim();
    output->showData();*/
}
    

void MaxPooling2x2Layer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    dedx->zeroAssign();
    //GPUMaxPoolingdx2x2v2(int* d_inputDim, float* d_input, int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy)
    GPUMaxPoolingdx2x2v2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, dedx->d_dimension, dedx->d_data, dedy->d_dimension, dedy->d_data);
}


//====================================================================================


//====================================================================================

MaxPooling2x2x2Layer::MaxPooling2x2x2Layer(){
    layerType = MAXPOOLING2X2X2;
}

void MaxPooling2x2x2Layer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) output->dimension[0]*output->dimension[1]*output->dimension[2]/32.0);
    BNy = ceil((float) output->dimension[3]/32.0);
    if((input->dimension[0] %2 != 0) || (input->dimension[1] %2 != 0) || (input->dimension[2] %2 != 0)){
        cout << "MaxPooling2x2x2Layer::setAll(): Max pooling dimension wrong.\n";
    }
}


void MaxPooling2x2x2Layer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUMaxPooling2x2x2v2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output)
    GPUMaxPooling2x2x2v2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, output->d_dimension, output->d_data);
}
    

void MaxPooling2x2x2Layer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    dedx->zeroAssign();
    //GPUMaxPoolingdx2x2x2v2(int* d_inputDim, float* d_input, int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy)
    GPUMaxPoolingdx2x2x2v2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, dedx->d_dimension, dedx->d_data, dedy->d_dimension, dedy->d_data);
}


//====================================================================================
  
    
    //========================================================
    MaxPooling3DLayer::MaxPooling3DLayer(){
    layerType = MAXPOOLING3D;
}

    void MaxPooling3DLayer::setFilterDim(int inDim1, int inDim2, int inDim3){
        filterDim1 = inDim1;
        filterDim2 = inDim2;
        filterDim3 = inDim3;
        xPadding = (filterDim1-1)/2;
        yPadding = (filterDim2-1)/2;
        zPadding = (filterDim3-1)/2;
    }
    
    void MaxPooling3DLayer::setAll(){
        checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) output->dimension[0]/8.0);
        BNy = ceil((float) output->dimension[1]/8.0);
        BNz = ceil((float) output->dimension[2]/8.0);
        nf = input->dimension[0]*input->dimension[1]*input->dimension[2];
        
        nb = output->dimension[0]*output->dimension[1]*output->dimension[2];
    }
    void MaxPooling3DLayer::forward(){
        checkAndCorrectInputDim();
        checkAndCorrectOutputDim();
    //GPUMaxPoolConv3D(float* d_in_x, float* d_out_y, 
    //                        int ThC, int ThR, int ThD, // thread number per col, row, depth
    //                        int filterXL, int filterYL, int filterZL, // filter length. 
    //                        int xPadding, int yPadding, int zPadding, // padding number
    //                        int outXL, int outYL) // output data length
        for(i=0; i<input->dimension[3]; i++){
            GPUMaxPoolConv3D<<<dim3(BNx, BNy, BNz), dim3(8,8,8)>>>(input->d_data + nf*i, 
                                                    output->d_data + nb*i, 
                                                    output->dimension[0], output->dimension[1], output->dimension[2],
                                                    filterDim1, filterDim2, filterDim3,
                                                    xPadding, yPadding, zPadding,
                                                    output->dimension[0], output->dimension[1]);
        }
    }
    void MaxPooling3DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    //GPUMaxPoolConv3Ddx(float* d_x, float* d_y, float* d_dedx, float* d_dedy,
    //                        int ThC, int ThR, int ThD, // thread number per col, row, depth
    //                        int filterXL, int filterYL, int filterZL, // filter length. 
    //                        int xPadding, int yPadding, int zPadding, // padding number
    //                        int outXL, int outYL) // output data length
        for(i=0; i<input->dimension[3]; i++){
            GPUMaxPoolConv3Ddx<<<dim3(BNx, BNy, BNz), dim3(8,8,8)>>>(input->d_data + nf*i, 
                                                        output->d_data + nb*i, 
                                                        dedx->d_data + nf*i, 
                                                        dedy->d_data + nb*i,
                                                        input->dimension[0], input->dimension[1], input->dimension[2],
                                                        filterDim1, filterDim2, filterDim3,
                                                        xPadding, yPadding, zPadding,
                                                        input->dimension[0], input->dimension[1]);
        }
    }
    //========================================================
    
    
    MaxPooling2DLayer::MaxPooling2DLayer(){
    layerType = MAXPOOLING2D;
}

    void MaxPooling2DLayer::setFilterDim(int inDim1, int inDim2){
        filterDim1 = inDim1;
        filterDim2 = inDim2;
        xPadding = (filterDim1-1)/2;
        yPadding = (filterDim2-1)/2;
    }
    
    void MaxPooling2DLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
        BNx = ceil((float) output->dimension[0]/16.0);
        BNy = ceil((float) output->dimension[1]/32.0);
        nf = input->dimension[0]*input->dimension[1];
        
        nb = output->dimension[0]*output->dimension[1];
    }
    void MaxPooling2DLayer::forward(){
        checkAndCorrectInputDim();
        checkAndCorrectOutputDim();
    //GPUMaxPoolConv2D(float* d_in_x, float* d_out_y, 
    //                        int ThC, int ThR, // thread number per col, row, depth
    //                        int filterXL, int filterYL, // filter length. 
    //                        int xPadding, int yPadding, // padding number
    //                        int outXL) // output data length
        for(i=0; i<input->dimension[2]; i++){
            GPUMaxPoolConv2D<<<dim3(BNx, BNy, 1), dim3(16,32,1)>>>(input->d_data + nf*i, 
                                                    output->d_data + nb*i, 
                                                    output->dimension[0], output->dimension[1], 
                                                    filterDim1, filterDim2, 
                                                    xPadding, yPadding, 
                                                    output->dimension[0]);
        }
    }
    void MaxPooling2DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    //GPUMaxPoolConv2Ddx(float* d_x, float* d_y, float* d_dedx, float* d_dedy,
    //                        int ThC, int ThR, // thread number per col, row, depth
    //                        int filterXL, int filterYL, // filter length. 
    //                        int xPadding, int yPadding, // padding number
    //                        int outXL) // output data length
        for(i=0; i<input->dimension[2]; i++){
            GPUMaxPoolConv2Ddx<<<dim3(BNx, BNy, 1), dim3(16,32,1)>>>(input->d_data + nf*i, 
                                                        output->d_data + nb*i, 
                                                        dedx->d_data + nf*i, 
                                                        dedy->d_data + nb*i,
                                                        input->dimension[0], input->dimension[1],
                                                        filterDim1, filterDim2,
                                                        xPadding, yPadding,
                                                        input->dimension[0]);
        }
    }
    //========================================================
    
    
    

ReLULayer::ReLULayer(){
    layerType = RELU;
}

    void ReLULayer::setAll(){
        dataLength = input->dataLength();
        BN = ceil((float) dataLength/1024.0);
    }
    void ReLULayer::forward(){
        /*cout << "ReLULayer::forward(): input dimension before check:\n";
        input->showDim();
        cout << "ReLULayer::forward(): output dimension before check:\n";
        input->showDim();*/
        checkAndCorrectInputDim();
        checkAndCorrectOutputDim();
        /*cout << "ReLULayer::forward(): input dimension after check:\n";
        output->showDim();
        cout << "ReLULayer::forward(): output dimension after check:\n";
        output->showDim();*/
        
        GPUReLU<<<BN, 1024>>>(input->d_data, output->d_data, dataLength);//float* d_in_x, float* d_out_y, int ThR)
    }
    void ReLULayer::backward(){
        checkAndCorrectdedxDim();
        checkAndCorrectdedyDim();
        GPUReLUdx<<<BN, 1024>>>(dedy->d_data, input->d_data, dedx->d_data, dataLength);//float* d_in_dedy, float* d_in_x, float* d_out_dedx, int ThR)
    }
    
FCLayer::FCLayer(){
    layerType = FC;
}

    void FCLayer::setAll(){
        weight = new dataTensor(output->dataLength(), input->dataLength());
        dedw = new dataTensor(output->dataLength(), input->dataLength());
        alldedw = new dataTensor(output->dataLength(), input->dataLength());
        alldedw->zeroAssign();
        BNf = ceil((float) output->dataLength()/1024.0);
        BNb = ceil((float) input->dataLength()/1024.0);
        BNx = ceil((float) input->dataLength()/32.0);
        BNy = ceil((float) output->dataLength()/32.0);
    }
    
    void FCLayer::forward(){
        GPUMProd<<<dim3(BNf, 1, 1), dim3(1024, 1, 1)>>>(weight->d_data, input->d_data, output->d_data, output->dataLength(), input->dataLength(), 1);
    }
    void FCLayer::backward(){
        GPUMProd<<<dim3(1, BNb, 1), dim3(1, 1024, 1)>>>(dedy->d_data, weight->d_data, dedx->d_data, 1, output->dataLength(), input->dataLength());
        tensorSum(alldedw, get_dedw(), 1);
    }
    
    dataTensor* FCLayer::get_dedw(){
        GPUFCcomputededw<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_data, dedy->d_data, dedw->d_data, input->dataLength(), output->dataLength());
        return(dedw);
    }
    
    
    void FCLayer::adjust(float alpha, float batchSize){
        GPUFCAdjust<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(weight->d_data, alldedw->d_data, input->dataLength(), output->dataLength(), alpha/batchSize);
        alldedw->zeroAssign();
    }
    
    
    
FCDOLayer::FCDOLayer(){
    layerType = FCDO;
    dropOutIdx = NULL;
    h_dropOutIdx = NULL;
}

    void FCDOLayer::setAll(){
        weight = new dataTensor(output->dataLength(), input->dataLength());
        dedw = new dataTensor(output->dataLength(), input->dataLength());
        alldedw = new dataTensor(output->dataLength(), input->dataLength());
        alldedw->zeroAssign();
        BNf = ceil((float) output->dataLength()/1024.0);
        BNb = ceil((float) input->dataLength()/1024.0);
        BNx = ceil((float) input->dataLength()/32.0);
        BNy = ceil((float) output->dataLength()/32.0);
        dropOutIdx = new dataTensor(output->dataLength());
        h_dropOutIdx = new float[output->dataLength()];
    }
    
    void FCDOLayer::forward(){
        int dropOutNum = (int) (dropOutRatio*((float) output->dataLength()) + 0.5);
        //cout << "FCDOLayer::forward() dropOutNum " << dropOutNum << endl;
        if (dropOutNum > 0){
            int* tempIdx = new int[dropOutNum];
            permutation(output->dataLength(), dropOutNum, tempIdx);
            for(i=0; i<output->dataLength(); i++){
                h_dropOutIdx[i] = 1;
            }
            for(i=0; i<dropOutNum; i++){
                h_dropOutIdx[tempIdx[i]] = 0;
            }
            dropOutIdx->resetData(h_dropOutIdx);
            delete[] tempIdx;
            tempIdx = NULL;
        }else{
            dropOutIdx->oneAssign();
        }
        
        GPUMProdDropOut<<<dim3(BNf, 1, 1), dim3(1024, 1, 1)>>>(weight->d_data, input->d_data, output->d_data, output->dataLength(), input->dataLength(), 1, dropOutIdx->d_data);
        
//        cout << "FCDOLayer::forward(): output\n";
//        output->showData();
        
    }
    void FCDOLayer::backward(){
        GPUMDropOutBack<<<dim3(1, BNb, 1), dim3(1, 1024, 1)>>>(dedy->d_data, weight->d_data, dedx->d_data, 1, output->dataLength(), input->dataLength(), dropOutIdx->d_data);
        tensorSum(alldedw, get_dedw(), 1);
    }
    
    dataTensor* FCDOLayer::get_dedw(){
        GPUFCcomputededwDropOut<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_data, dedy->d_data, dedw->d_data, input->dataLength(), output->dataLength(), dropOutIdx->d_data);
        return(dedw);
    }
    
    
    void FCDOLayer::adjust(float alpha, float batchSize){
        GPUFCAdjust<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(weight->d_data, alldedw->d_data, input->dataLength(), output->dataLength(), alpha/batchSize);
        alldedw->zeroAssign();
    }
    
    
    
FCBiasLayer::FCBiasLayer(){
    layerType = FCBIAS;
}

    void FCBiasLayer::setAll(){
        weight = new dataTensor(input->dataLength());
        BN = ceil((float) input->dataLength()/1024.0);
        dedw = new dataTensor(input->dataLength());
        alldedw = new dataTensor(input->dataLength());
        alldedw->zeroAssign();
    }
    void FCBiasLayer::forward(){
        GPUBias<<<BN, 1024>>>(input->d_data, output->d_data, weight->d_data, input->dataLength());
    }
    
    void FCBiasLayer::forward(dataTensor* dropOutIdx){
        GPUBiasDropOut<<<BN, 1024>>>(input->d_data, output->d_data, weight->d_data, input->dataLength(), dropOutIdx->d_data);
    }
    void FCBiasLayer::backward(){
        tensorSum(alldedw, get_dedw(), 1);
    }
    
    void FCBiasLayer::backward(dataTensor* dropOutIdx){
        tensorSum(alldedw, get_dedw(dropOutIdx), 1);
    }
    
    dataTensor* FCBiasLayer::get_dedw(dataTensor* dropOutIdx){
        GPUBiasdedw<<<BN, 1024>>>(dedw->d_data, dedy->d_data, dropOutIdx->d_data, input->dataLength());
        return(dedw);
    }
    dataTensor* FCBiasLayer::get_dedw(){
        return(dedy);
    }
    
    
    void FCBiasLayer::adjust(float alpha, float batchSize){
        GPUBiasAdjust<<<BN, 1024>>>(alldedw->d_data, weight->d_data, input->dataLength(), alpha/batchSize);
        alldedw->zeroAssign();
    }
    
    
//==============================================================================================

FCBRLayer::FCBRLayer(){
    layerType = FCBR;
    FC = NULL;
    bias = NULL;
    relu = NULL;
    h_dropOutIdx = NULL;
    dropOutIdx = NULL;
    FC2Bias = NULL;
    Bias2ReLU = NULL;
    Bias2FC = NULL;
    ReLU2Bias = NULL;
    FCWeight = NULL;
    BiasWeight = NULL;
    allWeight = NULL;
}

void FCBRLayer::setAll(){
    FC = new FCDOLayer;
    bias = new FCBiasLayer;
    relu = new ReLULayer;
    
    FC->setDropOutRatio(dropOutRatio);
    
    FC2Bias = new dataTensor(output->dataLength());
    Bias2ReLU = new dataTensor(output->dataLength());
    ReLU2Bias = new dataTensor(output->dataLength());
    //Bias2FC = new dataTensor(output->dataLength());
    
    outputDim = new int[1];
    outputDim[0] = output->dataLength();
    relu->setInputDim(1, outputDim);
    relu->setOutputDim(1, outputDim);
    delete[] outputDim;
    outputDim = NULL;
    
    
    FC->setInput(input);
    FC->setOutput(FC2Bias);
    FC->setdedx(dedx);
    FC->setdedy(ReLU2Bias);
    FC->setAll();
    
    bias->setInput(FC2Bias);
    bias->setOutput(Bias2ReLU);
    bias->setdedx(ReLU2Bias);
    bias->setdedy(ReLU2Bias);
    bias->setAll();
    
    relu->setInput(Bias2ReLU);
    relu->setOutput(output);
    relu->setdedx(ReLU2Bias);
    relu->setdedy(dedy);
    relu->setAll();
    
    FCWeight = FC->weight;
    BiasWeight = bias->weight;
}

void FCBRLayer::forward(){
    FC->forward();
    bias->forward(FC->dropOutIdx);
    relu->forward();
//    cout << "FCBRLayer::forward(): output\n";
//    relu->output->showData();
}

void FCBRLayer::backward(){
    relu->backward();
    bias->backward(FC->dropOutIdx);
    FC->backward();
}

void FCBRLayer::adjust(float alpha, float batchSize){
    FC->adjust(alpha, batchSize);
    bias->adjust(alpha, batchSize);
}

dataTensor* FCBRLayer::getWeight(){
    int weightLength = FCWeight->dataLength() + BiasWeight->dataLength();
    if (allWeight == NULL){
        //cout << "FCBRLayer::getWeight() Create allWeight.\n";
        allWeight = new dataTensor(weightLength);
    }
    float* h_weight = new float[weightLength];
    float* h_FCWeight = FCWeight->getData();
    float* h_BiasWeight = BiasWeight->getData();
    for(i=0; i<FCWeight->dataLength(); i++){
        h_weight[i] = h_FCWeight[i];
    }
    for(i=FCWeight->dataLength(); i<weightLength; i++){
        h_weight[i] = h_BiasWeight[i - FCWeight->dataLength()];
    }
    allWeight->resetData(h_weight);
    delete[] h_weight;
    delete[] h_FCWeight;
    delete[] h_BiasWeight;
    h_weight = NULL;
    return(allWeight);
}

void FCBRLayer::loadWeight(int dataLength, float* in_weight){
    int weightLength = FCWeight->dataLength() + BiasWeight->dataLength();
    if(weightLength == dataLength){
        FCWeight->resetData(in_weight);
        BiasWeight->resetData(in_weight + FCWeight->dataLength());
    }else{
        cout << "FCBRLayer::loadWeight() dataLength not consistent.\n";
    }
}

//==============================================================================================

LogSoftmaxLayer::LogSoftmaxLayer(){
    layerType = LOGSOFTMAX;
}

    void LogSoftmaxLayer::setAll(){
        classNumber = input->dataLength();
        h_dedx = new float[classNumber];
        expComp = new double[classNumber];
        h_in_x = new float[1];
    }
    void LogSoftmaxLayer::forward(){cout << "forward(): Please input the index of right class.\n";}
    float LogSoftmaxLayer::forward(int rightClass){
        delete[] h_in_x;
        h_in_x = NULL;
        h_in_x = input->getData();
        sum = 0.0;
        for(i=0; i<classNumber; i++){
            expComp[i] = exp((double) h_in_x[i]);
            sum += expComp[i];
        }
        loss = (float) (log(sum) - h_in_x[rightClass]);
        
        return(loss);
    }
    

    
    void LogSoftmaxLayer::backward(){cout << "backward(): Please input the index of right class.\n";}
    void LogSoftmaxLayer::backward(int rightClass){
        for(i=0; i<classNumber; i++){
            if(i != rightClass){
                    
                    
                
                h_dedx[i] = (float) expComp[i]/sum;
                
            }else{
                h_dedx[i] = -1 + (float) expComp[i]/sum;
            }
        }
        dedx->resetData(h_dedx);
    }
    void LogSoftmaxLayer::showDist(){
        cout << "The probability distribution is:\n";
        for(i=0; i<classNumber; i++){
             cout << (float) expComp[i]/sum << endl;
        }
        cout << endl;
    }
    //==========================
    
    diffSquareLayer::diffSquareLayer(){
    layerType = DIFFSQUARE;
}

    void diffSquareLayer::setAll(){
        classNumber = input->dataLength();
        h_dedx = new float[classNumber];
        expComp = new double[classNumber];
        h_in_x = new float[1];
    }
    void diffSquareLayer::forward(){cout << "forward(): Please input the index of right class.\n";}
    float diffSquareLayer::forward(int rightClass){
        delete[] h_in_x;
        h_in_x = NULL;
        h_in_x = input->getData();
        sum = 0.0;
        for(i=0; i<classNumber; i++){
            if(i != rightClass){
                expComp[i] = (double) h_in_x[i]*h_in_x[i];
            }else{
                expComp[i] = (double) (h_in_x[i]-1)*(h_in_x[i]-1);
            }
            
            sum += expComp[i];
        }
        loss = (float) sum/2.0;
        
        return(loss);
    }
    void diffSquareLayer::backward(){cout << "backward(): Please input the index of right class.\n";}
    void diffSquareLayer::backward(int rightClass){
        for(i=0; i<classNumber; i++){
            if(i != rightClass){
                
                    
                
                h_dedx[i] = h_in_x[i];
                
            }else{
                h_dedx[i] = -1 + h_in_x[i];
            }
        }
        dedx->resetData(h_dedx);
    }
    void diffSquareLayer::showDist(){
        cout << "The probability distribution is:\n";
        for(i=0; i<classNumber; i++){
             cout << (float) expComp[i]/sum << endl;
        }
        cout << endl;
    }
        //==========================
        
//==========================
    
    EucliDistLayer::EucliDistLayer(){
    layerType = EUCLIDIST;
}

    void EucliDistLayer::setAll(){
        tempDistance = new dataTensor(input->dimensionN, input->dimension);
        square = new dataTensor(input->dimensionN, input->dimension);
        tag = new dataTensor(input->dimensionN, input->dimension);
        loss = new dataTensor(1);
        loss->zeroAssign();
        BN = ceil((float) input->dataLength()/1024.0);
        
    }
    void EucliDistLayer::forward(){cout << "forward(): Please input the tag.\n";}
    float EucliDistLayer::forward(float* correctAns){
    
        tag->resetData(correctAns);
        // GPUDiffSquare(float* d_in_x, float* d_tag, float* d_diff, float* d_out_y, int dataLength)
        GPUDiffSquare<<<BN, 1024>>>(input->d_data, tag->d_data, tempDistance->d_data, square->d_data, input->dataLength());
        
        //cout << "EucliDistLayer::forward() tempDistance:\n";
        //tempDistance->showData();
        // GPUallAdd(float* d_x, float* d_y, int dataLength)
        GPUallAdd<<<BN, 1024>>>(square->d_data, loss->d_data, input->dataLength());
        //cout << "EucliDistLayer::forward() loss:\n";
        //loss->showData();
        
        float* temploss = loss->getData();
        float ans = temploss[0];
        delete[] temploss;
        temploss = NULL;
        loss->zeroAssign();
        return(ans);
    }
    void EucliDistLayer::backward(){
    
        //GPUScale(float* d_in, float* d_out, float scale, int length)
        GPUScale<<<BN, 1024>>>(tempDistance->d_data, dedx->d_data, 2.0, input->dataLength());
        
    }
    
        //==========================
        
//==========================
    
    naiveLossLayer::naiveLossLayer(){
    layerType = NAIVELOSS;
}

    void naiveLossLayer::setAll(){
        classNumber = input->dataLength();
        h_dedx = new float[classNumber];
        expComp = new double[classNumber];
        h_in_x = new float[1];
        subsOutput = new double[classNumber];
    }
    void naiveLossLayer::forward(){cout << "forward(): Please input the index of right class.\n";}
    float naiveLossLayer::forward(int rightClass){
        delete[] h_in_x;
        h_in_x = NULL;
        h_in_x = input->getData();
        
        minInd = 0;
        for(i=1; i<classNumber; i++){
            if(h_in_x[i] < h_in_x[minInd]){
                minInd = i;
            }
        }
        
        sum = 0.0;
        for(i=0; i<classNumber; i++){
            subsOutput[i] = (double) (h_in_x[i]-h_in_x[minInd]);
            sum += subsOutput[i];
        }
            
            
        
        loss = (float) 100.0*(1 - subsOutput[rightClass]/sum);
        
        return(loss);
    }
    void naiveLossLayer::backward(){cout << "backward(): Please input the index of right class.\n";}
    void naiveLossLayer::backward(int rightClass){
        double squareSum = sum*sum;
        for(i=0; i<classNumber; i++){
            if((i != rightClass) && (i != minInd)){
                h_dedx[i] = (float) 100.0*subsOutput[rightClass]/squareSum;
            }else if((i == rightClass) && (i != minInd)){
                h_dedx[i] = (float) 100.0*(subsOutput[rightClass] - sum)/squareSum;
            }else if((i != rightClass) && (i == minInd)){
                h_dedx[i] = (float) 100.0*(sum - (classNumber-1)*subsOutput[rightClass])/squareSum;
            }else{
                h_dedx[i] = 0.0;
            }
        }
        dedx->resetData(h_dedx);
    }
    void naiveLossLayer::showDist(){
    
    }
        //==========================
    

BindNtoOneLayer::BindNtoOneLayer(){
    layerType = BINDNTOONE;
}

void BindNtoOneLayer::add_input(dataTensor* in_x){
    x_bundle.push_back(in_x);
}
void BindNtoOneLayer::add_dedx(dataTensor* dedx){
    dedx_bundle.push_back(dedx);
}
  
void BindNtoOneLayer::setAll(){
    // record the input number, individual input length
    inputN = x_bundle.size();
    accLength = new int[inputN];
    accLength[0] = 0;
    for(i=1; i<inputN; i++){
        accLength[i] = x_bundle[i-1]->dataLength() + accLength[i-1];
    }
}

void BindNtoOneLayer::forward(){
    for(i=0; i<inputN; i++){
        cudaMemcpy(output->d_data + accLength[i], x_bundle[i]->d_data, x_bundle[i]->dataLength()*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void BindNtoOneLayer::backward(){
    for(i=0; i<inputN; i++){
        cudaMemcpy(dedx_bundle[i]->d_data, dedy->d_data + accLength[i], dedx_bundle[i]->dataLength()*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

//============================================================
StructBindLayer::StructBindLayer(){
    layerType = STRUCTBIND;
}

void StructBindLayer::add_input(dataTensor* in_x){
    x_bundle.push_back(in_x);
}
void StructBindLayer::add_dedx(dataTensor* dedx){
    dedx_bundle.push_back(dedx);
}
  
void StructBindLayer::setAll(){
    // record the input number, individual input length
    inputN = x_bundle.size();
    accLength = new int[inputN];
    accLength[0] = 0;
    for(i=1; i<inputN; i++){
        accLength[i] = x_bundle[i-1]->dataLength() + accLength[i-1];
    }
}

void StructBindLayer::forward(){
    for(i=0; i<inputN; i++){
        cudaMemcpy(output->d_data + accLength[i], x_bundle[i]->d_data, x_bundle[i]->dataLength()*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

void StructBindLayer::backward(){
    for(i=0; i<inputN; i++){
        cudaMemcpy(dedx_bundle[i]->d_data, dedy->d_data + accLength[i], dedx_bundle[i]->dataLength()*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}
//============================================================


SeparateOnetoNLayer::SeparateOnetoNLayer(){
    layerType = SEPARATEONETON;
}

void SeparateOnetoNLayer::setAll(){
    outputN = dedy_bundle.size();
    BN = ceil((float) dedy_bundle[0]->dataLength()/512.0);
    
    h_dedyHeads = new float*[outputN];
    for(i=0; i<outputN; i++){
        h_dedyHeads[i] = dedy_bundle[i]->d_data;
    }
    cudaError_t errmsg;
    
    errmsg = cudaMalloc((void ***) &d_dedyHeads, outputN*sizeof(float*));
    if(errmsg == cudaErrorMemoryAllocation){
        cout << "SeparateOnetoNLayer::setAll() cudaMalloc Error!\n";
        ALLBREAK = 1;
    }
    cudaMemcpy(d_dedyHeads, h_dedyHeads, outputN*sizeof(float*), cudaMemcpyHostToDevice);
}


void SeparateOnetoNLayer::add_dedy(dataTensor* dedy){
    dedy_bundle.push_back(dedy);
}

void SeparateOnetoNLayer::forward(){
// This function is not necessary. Just assign pointer y to pointer x of each layer outside.
}

void SeparateOnetoNLayer::backward(){
// GPUVectorSum computes sum_{i=1}^vectorN d_dataHeads[i].
//__global__ void GPUVectorSum(float** d_dataHeads, float* d_out, int vectorN, int dataL)
    
    GPUVectorSum<<<BN, 512>>>(d_dedyHeads, dedx->d_data, outputN, dedy_bundle[0]->dataLength());
}

SumNtoOneLayer::SumNtoOneLayer(){
    layerType = SUMNTOONE;
}

void SumNtoOneLayer::add_input(dataTensor* in_x){
    if(x_bundle.size() == 1){
        x_bundleDimN = x_bundle[0]->dimensionN;
        x_bundleDim = x_bundle[0]->dimension;
    }
    if(x_bundle.size() > 0){
        if(in_x->dimensionN == x_bundleDimN){
            for(i=0; i<x_bundleDimN-1; i++){
                if(in_x->dimension[i] != x_bundleDim[i]){
                    cout << "Size of input x wrong!\n";
                    i = x_bundleDimN;
                }
            }
        }else{
            cout << "Dimension of input x wrong!\n";
        }
    }
    x_bundle.push_back(in_x);
}
void SumNtoOneLayer::add_dedx(dataTensor* dedx){
    if(dedx_bundle.size() == 1){
        x_bundleDimN = dedx_bundle[0]->dimensionN;
        x_bundleDim = dedx_bundle[0]->dimension;
    }
    if(dedx_bundle.size() > 0){
        if(dedx->dimensionN == x_bundleDimN){
            for(i=0; i<x_bundleDimN-1; i++){
                if(dedx->dimension[i] != x_bundleDim[i]){
                    cout << "Size of dedx wrong!\n";
                    i = x_bundleDimN;
                }
            }
        }else{
            cout << "Dimension of dedx wrong!\n";
        }
    }
    dedx_bundle.push_back(dedx);
}

void SumNtoOneLayer::setAll(){// Call this function only after adding all x and dedx.
    
    inputN = x_bundle.size();
    
    featureLength = 1;
    for(i=0; i<x_bundleDimN-1; i++){
        featureLength *= x_bundleDim[i];
    }
    
    bundleTotalLength = 0;
    for(i=0; i<x_bundle.size(); i++){
        bundleTotalLength += featureLength*(x_bundle[i]->dimension[x_bundleDimN-1]);
    }
    bundleTotalFeatureN = bundleTotalLength/featureLength;
    
    h_xHeads = new float*[bundleTotalFeatureN];
    h_dedxHeads = new float*[bundleTotalFeatureN];
    
    int headInd = 0;
    for(i=0; i<inputN; i++){
        for(int j=0; j<x_bundle[i]->dimension[x_bundleDimN-1]; j++){
            h_xHeads[headInd] = x_bundle[i]->d_data + j*featureLength;
            h_dedxHeads[headInd] = dedx_bundle[i]->d_data + j*featureLength;
            headInd++;
        }
        
    }
    cudaError_t errmsg;
    errmsg = cudaMalloc((void ***) &d_xHeads, bundleTotalLength*sizeof(float*));
    if(errmsg == cudaErrorMemoryAllocation){
        cout << "SumNtoOneLayer::setAll() cudaMalloc Error!\n";
        ALLBREAK = 1;
    }
    cudaMemcpy(d_xHeads, h_xHeads, bundleTotalLength*sizeof(float*), cudaMemcpyHostToDevice);
    
    BN = ceil((float) featureLength/512.0);
}

void SumNtoOneLayer::forward(){
                            //float** d_dataHeads, float* d_out, int vectorN, int dataL
    GPUVectorSum<<<BN, 512>>>(d_xHeads, output->d_data, bundleTotalFeatureN, featureLength);
}
void SumNtoOneLayer::backward(){
    for(i=0; i<bundleTotalFeatureN; i++){
        cudaMemcpy(h_dedxHeads[i], dedy->d_data, featureLength*sizeof(float), cudaMemcpyDeviceToDevice);
    }
}

//====================================================================================

NormalizeLayer::NormalizeLayer(){
    layerType = NORMALIZE;
    scale = 1.0;
}

NormalizeLayer::NormalizeLayer(float in_scale){
    layerType = NORMALIZE;
    scale = in_scale;
}

void NormalizeLayer::setAll(){
    BN = ceil((float) input->dataLength()/1024.0);
}

void NormalizeLayer::forward(){
    float fraction;
    float* temp = input->getData();
    maxInd = 0;
    maxVal = (float) fabs((double) temp[0]);
    
    
    for(i=1; i<input->dataLength(); i++){
        if(maxVal < fabs((double) temp[i])){
            maxInd = i;
            maxVal = (float) fabs((double) temp[i]);
        }
    }
    
//    cout << "NormalizeLayer::forward(): max is " << maxVal << endl;
    
    if(temp[maxInd]>=0){
        positive = 1;
    }else{
        positive = -1;
    }
    
    if(maxVal == 0.0){
//        cout << "NormalizeLayer::forward(): max is zero!" << endl;
        maxVal = 1;
    }
    
    //fraction = 1.0/maxVal;
    //fraction = 10.0/maxVal; // scale up ten times
    fraction = scale/maxVal;
    
    //GPUScale(float* d_in, float* d_out, float scale, int length)
    //cout << "NormalizeLayer::forward(): dataLength is " << input->dataLength() << endl;
    //cout << "NormalizeLayer::forward(): maxVal is " << maxVal << endl;
    
    GPUScale<<<BN, 1024>>>(input->d_data, output->d_data, fraction, input->dataLength());
    delete[] temp;
    temp = NULL;
    
    /*cout << "NormalizeLayer::forward(): id: " << this->id << ", input data length: " << endl;
    cout << input->dataLength() << endl;
    cout << "NormalizeLayer::forward(): id: " << this->id << ", output: " << endl;
    output->showDim();
    output->showData();*/
}

void NormalizeLayer::backward(){
/*    cout << "NormalizeLayer::backward(): input->dataLength() " << input->dataLength() << endl;
    cout << "NormalizeLayer::backward(): dedx->dataLength() " << dedx->dataLength() << endl;
    cout << "NormalizeLayer::backward(): dedy->dataLength() " << dedy->dataLength() << endl;
    
    cout << "NormalizeLayer::backward(): input->d_data " << endl;
    input->showData(); 
    cout << "NormalizeLayer::backward(): output->d_data " << endl;
    output->showData(); 
    cout << "NormalizeLayer::backward(): dedx->d_data " << endl;
    dedx->showData(); 
    cout << "NormalizeLayer::backward(): dedy->d_data " << endl;
    dedy->showData(); */
    
    
    //cudaMemcpy(dedx->d_data, dedy->d_data, (input->dataLength())*sizeof(float), cudaMemcpyDeviceToDevice);
    GPUNormalBack<<<BN, 1024>>>(dedx->d_data, dedy->d_data, input->d_data, maxInd, maxVal, input->dataLength(), positive, scale);
    //GPUScale<<<BN, 512>>>(dedx->d_data, dedx->d_data, scale, input->dataLength()); //scale up ten times
    //(float* d_dedx, float* d_dedy, float* d_x, int maxInd, float maxVal, int length, int positive)
}



//==================================================================================

ZeroPad1DLayer::ZeroPad1DLayer(){
    layerType = ZEROPADDING1D;
}

void ZeroPad1DLayer::setPadding(int x1, int x2){
    x_padding1 = x1;
    x_padding2 = x2;
    output->zeroAssign();
}

void ZeroPad1DLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) input->dimension[0]/32.0);
    BNy = ceil((float) input->dimension[1]/32.0);
    in_featureN = input->dimension[1];
}

void ZeroPad1DLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    GPUZeroPad1Dv2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, output->d_dimension, output->d_data, x_padding1);
}

void ZeroPad1DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    GPUZeroPadBack1Dv2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(dedx->d_dimension, dedx->d_data, dedy->d_dimension, dedy->d_data, x_padding1);
}

//==================================================================================



//==================================================================================

ZeroPad2DLayer::ZeroPad2DLayer(){
    layerType = ZEROPADDING2D;
}

void ZeroPad2DLayer::setPadding(int x1, int x2, int y1, int y2){
    x_padding1 = x1;
    x_padding2 = x2;
    y_padding1 = y1;
    y_padding2 = y2;
    output->zeroAssign();
}

void ZeroPad2DLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) input->dimension[0]*input->dimension[1]/32.0);
    BNy = ceil((float) input->dimension[2]/32.0);
    in_featureN = input->dimension[2];
}

void ZeroPad2DLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    //GPUZeroPad2Dv2(int* d_inputDim, float* d_input, int* d_outputDim, float* d_output, int x1, int y1)
    GPUZeroPad2Dv2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, output->d_dimension, output->d_data, x_padding1, y_padding1);
    
    /*cout << "ZeroPad2DLayer::forward(): id: " << this->id << ", input:\n";
    cout << "input data length: " << input->dataLength();
    input->showDim();
    input->showData();
    cout << "ZeroPad2DLayer::forward(): id: " << this->id << ", output:\n";
    cout << "output data length: " << output->dataLength();
    output->showDim();
    output->showData();*/
}

void ZeroPad2DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    //GPUZeroPadBack2Dv2(int* d_dedxDim, float* d_dedx, int* d_dedyDim, float* d_dedy, int x1, int y1)
    GPUZeroPadBack2Dv2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(dedx->d_dimension, dedx->d_data, dedy->d_dimension, dedy->d_data, x_padding1, y_padding1);
}

//==================================================================================

ZeroPad3DLayer::ZeroPad3DLayer(){
    layerType = ZEROPADDING3D;
}

void ZeroPad3DLayer::setPadding(int x1, int x2, int y1, int y2, int z1, int z2){
    x_padding1 = x1;
    x_padding2 = x2;
    y_padding1 = y1;
    y_padding2 = y2;
    z_padding1 = z1;
    z_padding2 = z2;
    output->zeroAssign();
}

void ZeroPad3DLayer::setAll(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    BNx = ceil((float) input->dimension[0]*input->dimension[1]*input->dimension[2]/32.0);
    BNy = ceil((float) input->dimension[3]/32.0);
    in_featureN = input->dimension[3];
}

void ZeroPad3DLayer::forward(){
    checkAndCorrectInputDim();
    checkAndCorrectOutputDim();
    GPUZeroPad3Dv2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(input->d_dimension, input->d_data, output->d_dimension, output->d_data, x_padding1, y_padding1, z_padding1);
}

void ZeroPad3DLayer::backward(){
    checkAndCorrectdedxDim();
    checkAndCorrectdedyDim();
    GPUZeroPadBack3Dv2<<<dim3(BNx, BNy, 1), dim3(32, 32, 1)>>>(dedx->d_dimension, dedx->d_data, dedy->d_dimension, dedy->d_data, x_padding1, y_padding1, z_padding1);
}


//==================================================================================

inception3DPack::inception3DPack(int batchSize, int in_oneConv1FMN, int in_oneConv2FMN, int in_oneConv3FMN, int in_oneConv4FMN, int in_threeConvFMN, int in_fiveConvFMN){
    fraction = 1.0/(float) batchSize;
    
    oneConv1FMN = in_oneConv1FMN;
    oneConv2FMN = in_oneConv2FMN;
    oneConv3FMN = in_oneConv3FMN;
    oneConv4FMN = in_oneConv4FMN;
    threeConvFMN = in_threeConvFMN;
    fiveConvFMN = in_fiveConvFMN;
    
    inputLayer = new SeparateOnetoNLayer;
    
    oneConv1 = new Conv3DLayer;
    oneConv2 = new Conv3DLayer;
    oneConv3 = new Conv3DLayer;
    oneConv4 = new Conv3DLayer;
    threeConv = new Conv3DLayer;
    fiveConv = new Conv3DLayer;
    threeMaxPooling = new MaxPooling3DLayer;
    
    afterOneConv1 = new ReLULayer;
    afterOneConv2 = new ReLULayer;
    afterOneConv3 = new ReLULayer;
    afterOneConv4 = new ReLULayer;
    afterThreeConv = new ReLULayer;
    afterFiveConv = new ReLULayer;
    
    outputLayer = new StructBindLayer;
}

void inception3DPack::setInput(dataTensor* x){
    inputLayer->setInput(x);
}

void inception3DPack::setOutput(dataTensor* y){
    outputLayer->setOutput(y);
}

void inception3DPack::setdedx(dataTensor* dedx){
    inputLayer->setdedx(dedx);
}

void inception3DPack::setdedy(dataTensor* dedy){
    outputLayer->setdedy(dedy);
}

void inception3DPack::setAll(){
    dimensionN = inputLayer->input->dimensionN;
    dimension = inputLayer->input->dimension;
    
    
    dataTensor* oneConv1x;
    dataTensor* oneConv2x;
    dataTensor* oneConv3x;
    dataTensor* oneConv4x;
    dataTensor* threeConvx;
    dataTensor* fiveConvx;
    dataTensor* threeMaxPoolingx;
    dataTensor* relux1; // after oneConv1
    dataTensor* relux2; // after oneConv2
    dataTensor* relux3; // after oneConv3
    dataTensor* relux4; // after oneConv4
    dataTensor* relux5; // after threeConv
    dataTensor* relux6; // after fiveConv
    
    
    dataTensor* oneConv1y;
    dataTensor* oneConv2y;
    dataTensor* oneConv3y;
    dataTensor* oneConv4y;
    dataTensor* threeConvy;
    dataTensor* fiveConvy;
    dataTensor* threeMaxPoolingy;
    dataTensor* reluy1; // after oneConv1
    dataTensor* reluy2; // after oneConv2
    dataTensor* reluy3; // after oneConv3
    dataTensor* reluy4; // after oneConv4
    dataTensor* reluy5; // after threeConv
    dataTensor* reluy6; // after fiveConv
    
    
    dataTensor* oneConv1dedx;
    dataTensor* oneConv2dedx;
    dataTensor* oneConv3dedx;
    dataTensor* oneConv4dedx;
    dataTensor* threeConvdedx;
    dataTensor* fiveConvdedx;
    dataTensor* threeMaxPoolingdedx;
    dataTensor* reludedx1; // after oneConv1
    dataTensor* reludedx2; // after oneConv2
    dataTensor* reludedx3; // after oneConv3
    dataTensor* reludedx4; // after oneConv4
    dataTensor* reludedx5; // after threeConv
    dataTensor* reludedx6; // after fiveConv
    
    
    dataTensor* oneConv1dedy;
    dataTensor* oneConv2dedy;
    dataTensor* oneConv3dedy;
    dataTensor* oneConv4dedy;
    dataTensor* threeConvdedy;
    dataTensor* fiveConvdedy;
    dataTensor* threeMaxPoolingdedy;
    dataTensor* reludedy1; // after oneConv1
    dataTensor* reludedy2; // after oneConv2
    dataTensor* reludedy3; // after oneConv3
    dataTensor* reludedy4; // after oneConv4
    dataTensor* reludedy5; // after threeConv
    dataTensor* reludedy6; // after fiveConv
    
    
    oneConv1x = inputLayer->input;
    oneConv2x = inputLayer->input;
    oneConv3x = inputLayer->input;
    threeMaxPoolingx = inputLayer->input;
    
    
    oneConv1y = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv1FMN);
    oneConv2y = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv2FMN);
    oneConv3y = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv3FMN);
    threeMaxPoolingy = new dataTensor(dimensionN, dimension);
    
    
    relux1 = oneConv1y;
    relux2 = oneConv2y;
    relux3 = oneConv3y;
    oneConv4x = threeMaxPoolingy;
    
    
    
    reluy1 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv1FMN);
    reluy2 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv2FMN);
    reluy3 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv3FMN);
    
    threeConvx = reluy2;
    fiveConvx = reluy3;
    
    threeConvy = new dataTensor(dimension[0], dimension[1], dimension[2], threeConvFMN);
    fiveConvy = new dataTensor(dimension[0], dimension[1], dimension[2], fiveConvFMN);
    oneConv4y = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv4FMN);
    
    relux5 = threeConvy;
    relux6 = fiveConvy;
    relux4 = oneConv4y;
    
    reluy5 = new dataTensor(dimension[0], dimension[1], dimension[2], threeConvFMN);
    reluy6 = new dataTensor(dimension[0], dimension[1], dimension[2], fiveConvFMN);
    reluy4 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv4FMN);
    
    reludedy1 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv1FMN);
    reludedy5 = new dataTensor(dimension[0], dimension[1], dimension[2], threeConvFMN);
    reludedy6 = new dataTensor(dimension[0], dimension[1], dimension[2], fiveConvFMN);
    reludedy4 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv4FMN);
    
    reludedx1 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv1FMN);
    reludedx5 = new dataTensor(dimension[0], dimension[1], dimension[2], threeConvFMN);
    reludedx6 = new dataTensor(dimension[0], dimension[1], dimension[2], fiveConvFMN);
    reludedx4 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv4FMN);
    
    oneConv1dedy = reludedx1;
    threeConvdedy = reludedx5;
    fiveConvdedy = reludedx6;
    oneConv4dedy = reludedx4;
    
    oneConv1dedx = new dataTensor(dimension[0], dimension[1], dimension[2], dimension[3]);
    threeConvdedx = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv2FMN);
    fiveConvdedx = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv3FMN);
    oneConv4dedx = new dataTensor(dimension[0], dimension[1], dimension[2], dimension[3]);
    
    reludedy2 = threeConvdedx;
    reludedy3 = fiveConvdedx;
    
    reludedx2 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv2FMN);
    reludedx3 = new dataTensor(dimension[0], dimension[1], dimension[2], oneConv3FMN);
    
    oneConv2dedy = reludedx2;
    oneConv3dedy = reludedx3;
    threeMaxPoolingdedy = oneConv4dedx;
    
    oneConv2dedx = new dataTensor(dimensionN, dimension);
    oneConv3dedx = new dataTensor(dimensionN, dimension);
    threeMaxPoolingdedx = new dataTensor(dimensionN, dimension);
    
    
    oneConv1->setInput(oneConv1x);
    oneConv1->setOutput(oneConv1y);
    oneConv1->setdedx(oneConv1dedx);
    oneConv1->setdedy(oneConv1dedy);
    
    oneConv2->setInput(oneConv2x);
    oneConv2->setOutput(oneConv2y);
    oneConv2->setdedx(oneConv2dedx);
    oneConv2->setdedy(oneConv2dedy);
    
    oneConv3->setInput(oneConv3x);
    oneConv3->setOutput(oneConv3y);
    oneConv3->setdedx(oneConv3dedx);
    oneConv3->setdedy(oneConv3dedy);
    
    oneConv4->setInput(oneConv4x);
    oneConv4->setOutput(oneConv4y);
    oneConv4->setdedx(oneConv4dedx);
    oneConv4->setdedy(oneConv4dedy);
    
    threeMaxPooling->setInput(threeMaxPoolingx);
    threeMaxPooling->setOutput(threeMaxPoolingy);
    threeMaxPooling->setdedx(threeMaxPoolingdedx);
    threeMaxPooling->setdedy(threeMaxPoolingdedy);
    
    threeConv->setInput(threeConvx);
    threeConv->setOutput(threeConvy);
    threeConv->setdedx(threeConvdedx);
    threeConv->setdedy(threeConvdedy);
    
    fiveConv->setInput(fiveConvx);
    fiveConv->setOutput(fiveConvy);
    fiveConv->setdedx(fiveConvdedx);
    fiveConv->setdedy(fiveConvdedy);
    
    afterOneConv1->setInput(relux1);
    afterOneConv1->setOutput(reluy1);
    afterOneConv1->setdedx(reludedx1);
    afterOneConv1->setdedy(reludedy1);
    
    afterOneConv2->setInput(relux2);
    afterOneConv2->setOutput(reluy2);
    afterOneConv2->setdedx(reludedx2);
    afterOneConv2->setdedy(reludedy2);
    
    afterOneConv3->setInput(relux3);
    afterOneConv3->setOutput(reluy3);
    afterOneConv3->setdedx(reludedx3);
    afterOneConv3->setdedy(reludedy3);
    
    afterOneConv4->setInput(relux4);
    afterOneConv4->setOutput(reluy4);
    afterOneConv4->setdedx(reludedx4);
    afterOneConv4->setdedy(reludedy4);
    
    afterThreeConv->setInput(relux5);
    afterThreeConv->setOutput(reluy5);
    afterThreeConv->setdedx(reludedx5);
    afterThreeConv->setdedy(reludedy5);
    
    afterFiveConv->setInput(relux6);
    afterFiveConv->setOutput(reluy6);
    afterFiveConv->setdedx(reludedx6);
    afterFiveConv->setdedy(reludedy6);
    
    
    inputLayer->add_dedy(oneConv1dedx);
    inputLayer->add_dedy(oneConv2dedx);
    inputLayer->add_dedy(oneConv3dedx);
    inputLayer->add_dedy(threeMaxPoolingdedx);
    
    outputLayer->add_input(reluy1);
    outputLayer->add_input(reluy5);
    outputLayer->add_input(reluy6);
    outputLayer->add_input(reluy4);
    
    outputLayer->add_dedx(reludedy1);
    outputLayer->add_dedx(reludedy5);
    outputLayer->add_dedx(reludedy6);
    outputLayer->add_dedx(reludedy4);
    
    dataTensor* oneConv1W = new dataTensor(1, 1, 1, dimension[3], oneConv1FMN);
    dataTensor* oneConv2W = new dataTensor(1, 1, 1, dimension[3], oneConv2FMN);
    dataTensor* oneConv3W = new dataTensor(1, 1, 1, dimension[3], oneConv3FMN);
    dataTensor* oneConv4W = new dataTensor(1, 1, 1, dimension[3], oneConv4FMN);
    dataTensor* threeConvW = new dataTensor(3, 3, 3, oneConv2FMN, threeConvFMN);
    dataTensor* fiveConvW = new dataTensor(5, 5, 5, oneConv3FMN, fiveConvFMN);
    
    ((ProtoLayer*) oneConv1)->setWeight(oneConv1W);
    ((ProtoLayer*) oneConv2)->setWeight(oneConv2W);
    ((ProtoLayer*) oneConv3)->setWeight(oneConv3W);
    ((ProtoLayer*) oneConv4)->setWeight(oneConv4W);
    ((ProtoLayer*) threeConv)->setWeight(threeConvW);
    ((ProtoLayer*) fiveConv)->setWeight(fiveConvW);
    
    threeMaxPooling->setFilterDim(3, 3, 3);
    
    inputLayer->setAll();
    oneConv1->setAll();
    oneConv2->setAll();
    oneConv3->setAll();
    oneConv4->setAll();
    threeConv->setAll();
    fiveConv->setAll();
    afterOneConv1->setAll();
    afterOneConv2->setAll();
    afterOneConv3->setAll();
    afterOneConv4->setAll();
    afterThreeConv->setAll();
    afterFiveConv->setAll();
    threeMaxPooling->setAll();
    outputLayer->setAll();
    
    oneConv1Alldedw = new dataTensor(1, 1, 1, dimension[3], oneConv1FMN);
    oneConv2Alldedw = new dataTensor(1, 1, 1, dimension[3], oneConv2FMN);
    oneConv3Alldedw = new dataTensor(1, 1, 1, dimension[3], oneConv3FMN);
    oneConv4Alldedw = new dataTensor(1, 1, 1, dimension[3], oneConv4FMN);
    threeConvAlldedw = new dataTensor(3, 3, 3, oneConv2FMN, threeConvFMN);
    fiveConvAlldedw = new dataTensor(5, 5, 5, oneConv3FMN, fiveConvFMN);
    
    oneConv1Alldedw->zeroAssign();
    oneConv2Alldedw->zeroAssign();
    oneConv3Alldedw->zeroAssign();
    oneConv4Alldedw->zeroAssign();
    threeConvAlldedw->zeroAssign();
    fiveConvAlldedw->zeroAssign();
    
    
    
}


void inception3DPack::forward(){
    oneConv1->forward();
    oneConv2->forward();
    oneConv3->forward();
    threeMaxPooling->forward();
    afterOneConv1->forward();
    afterOneConv2->forward();
    afterOneConv3->forward();
    threeConv->forward();
    fiveConv->forward();
    oneConv4->forward();
    afterThreeConv->forward();
    afterFiveConv->forward();
    afterOneConv4->forward();
    outputLayer->forward();
}

void inception3DPack::backward(){
    outputLayer->backward();
    afterOneConv4->backward();
    afterFiveConv->backward();
    afterThreeConv->backward();
    oneConv4->backward();
    fiveConv->backward();
    threeConv->backward();
    afterOneConv3->backward();
    afterOneConv2->backward();
    afterOneConv1->backward();
    threeMaxPooling->backward();
    oneConv3->backward();
    oneConv2->backward();
    oneConv1->backward();
    inputLayer->backward();
    
}

void inception3DPack::sumUp_dedw(){
    dataTensor* tempTensor;
    tempTensor = oneConv1->get_dedw();
    tensorSum(oneConv1Alldedw, tempTensor, fraction);
    //delete tempTensor; Never delete tempTensor, because get_dedw() returns the pointer of the internal dataTensor.
        
    tempTensor = oneConv2->get_dedw();
    tensorSum(oneConv2Alldedw, tempTensor, fraction);
    
    tempTensor = oneConv3->get_dedw();
    tensorSum(oneConv3Alldedw, tempTensor, fraction);
    
    tempTensor = oneConv4->get_dedw();
    tensorSum(oneConv4Alldedw, tempTensor, fraction);
    
    tempTensor = threeConv->get_dedw();
    tensorSum(threeConvAlldedw, tempTensor, fraction);
    
    tempTensor = fiveConv->get_dedw();
    tensorSum(fiveConvAlldedw, tempTensor, fraction);
}

void inception3DPack::adjust(float alpha){
/*
    oneConv1->adjust(alpha, oneConv1Alldedw);
    oneConv2->adjust(alpha, oneConv2Alldedw);
    oneConv3->adjust(alpha, oneConv3Alldedw);
    oneConv4->adjust(alpha, oneConv4Alldedw);
    threeConv->adjust(alpha, threeConvAlldedw);
    fiveConv->adjust(alpha, fiveConvAlldedw);
    
    oneConv1Alldedw->zeroAssign();
    oneConv2Alldedw->zeroAssign();
    oneConv3Alldedw->zeroAssign();
    oneConv4Alldedw->zeroAssign();
    threeConvAlldedw->zeroAssign();
    fiveConvAlldedw->zeroAssign();
    */
}

