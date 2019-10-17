#include <vector>
#include <string>
#include "LRP.h"
#include "kernel3.h"
#include <iostream>
#include <math.h>

using namespace std;

MatStruct LRP_LOGSOFTMAX(LogSoftmaxLayer* layerBody, int tag){
    cout << "Processing layer " << layerBody->id << endl;
    int classNumber = layerBody->classNumber;
    double* expComp = layerBody->expComp;
    MatStruct outLRP(classNumber);
    outLRP.zeroAssign();
    double score = 0.0;
    for (int i = 0; i < classNumber; i++){
        score += expComp[i];
    }
    score = expComp[tag]/score;
    outLRP.set(score, tag);

    return outLRP;
}

MatStruct LRP_CBR1D(CBR1DLayer* layerBody, MatStruct inLRP){
    cout << "Processing layer " << layerBody->id << endl;
    // Divied into 3 steps. inLRP -> ReLU, ReLU -> Bias, Bias -> convolution.
    float zeroError = 0.00001; // Numbers less than zeroError are treated as zero.

    // inLRP -> ReLU
    MatStruct output = layerBody->relu->output->getMatStruct();

    // for debug
    // cout << "LRP_CBR1D(): relu output dim: ";
    // output.showDim();

    inLRP.reshape(output.size());
    // set LRP zero if output of ReLU is zero
    long long int dataLength = output.getLength();
    long long int i = 0; // loop index
    for (i = 0; i < dataLength; i++){
        if (output.get(i) < zeroError){
            inLRP.set(0.0, i);
        }
    }

    // ReLU -> Bias
    output = layerBody->bias->output->getMatStruct(); 

    // for debug
    // cout << "LRP_CBR1D(): bias output dim: ";
    // output.showDim();

    // Do nothing if bias is used to calculate LRP.
    // Otherwise, output of bias -= bias is needed to remove bias.
    
    
    // Bias -> convolution
    MatStruct weight = layerBody->conv->weight->getMatStruct(); 
    MatStruct input = layerBody->conv->input->getMatStruct();

    // for debug
    // cout << "LRP_CBR1D(): conv input dim: ";
    // input.showDim();
    // cout << "LRP_CBR1D(): conv weight dim: ";
    // weight.showDim();

    vector<int> inputDim = input.size(); // Dimension of input should be n x inChannel, where n is the dimension of a single channel (or feature map).
    vector<int> weightDim = weight.size(); // Dimension of weight should be n x inChannel x outChannel, where n is the filter dimension.
    vector<int> outputDim = output.size(); // Dimension of output should be n x outChannel, where n is the dimension of a single channel (or feature map).
    // inputDim[0] should be the same with outputDim[0];
    int filterDim = weightDim[0];
    int paddingNum = (filterDim - 1)/2;
    int inChannelNum = inputDim[1];
    int outChannelNum = outputDim[1];
    int dataDim = inputDim[0]; // Modify this to a vector if data is multi-dimensional.
    MatStruct outLRP(inputDim);
    outLRP.zeroAssign();

    int inChannel_idx = 0;
    int outChannel_idx = 0;
    int in_data_idx = 0;
    int out_data_idx = 0;
    /*
        i: in_data_idx
        j: out_data_idx
        l: paddingNum
        wl: weight length
        x: input data
        y: output data
        w: weight
        Rx: outLRP
        Ry: inLRP
        Rx_i = x_i * sum_j (Ry_j * w_{i - j + l}/y_j), j ranges from i + l to i + l - wl + 1.
        This equation does not expicitly describe the channels.
    */
    for (inChannel_idx = 0; inChannel_idx < inChannelNum; inChannel_idx++){
        for (in_data_idx = 0; in_data_idx < dataDim; in_data_idx++){
            float temp = 0;
            for (outChannel_idx = 0; outChannel_idx < outChannelNum; outChannel_idx++){
                for (out_data_idx = in_data_idx + paddingNum - weightDim[0] + 1; out_data_idx <= in_data_idx + paddingNum; out_data_idx++){
                    if(out_data_idx >= 0 && out_data_idx < outputDim[0]){
                        // for debug
                        //cout << "inChannel_idx: " << inChannel_idx << ", in_data_idx: " << in_data_idx << ", outChannel_idx: " << outChannel_idx << ", out_data_idx: " << out_data_idx << endl;
                        if (output.get(out_data_idx, outChannel_idx) > zeroError || output.get(out_data_idx, outChannel_idx) < -zeroError){

                            // for debug
                            // cout << "inLRP.get(out_data_idx, outChannel_idx)" << endl;
                            // inLRP.get(out_data_idx, outChannel_idx);
                            // cout << "weight.get(in_data_idx - out_data_idx + paddingNum, inChannel_idx, outChannel_idx)" << endl;
                            // weight.get(in_data_idx - out_data_idx + paddingNum, inChannel_idx, outChannel_idx);
                            // cout << "output.get(out_data_idx, outChannel_idx)" << endl;
                            // output.get(out_data_idx, outChannel_idx);
                            // cout << endl;

                            temp += inLRP.get(out_data_idx, outChannel_idx) * weight.get(in_data_idx - out_data_idx + paddingNum, inChannel_idx, outChannel_idx)/output.get(out_data_idx, outChannel_idx);
                        }
                    }
                        
                }
            }
            temp *= input.get(in_data_idx, inChannel_idx);
            outLRP.set(temp, in_data_idx, inChannel_idx);
        }
    }

    return outLRP;
}

MatStruct LRP_MAXPOOLING2(MaxPooling2DLayer* layerBody, MatStruct inLRP){
    cout << "Processing layer " << layerBody->id << endl;
    MatStruct input = layerBody->input->getMatStruct();
    MatStruct output = layerBody->output->getMatStruct();
    inLRP.reshape(output.size());

    // for debug
    // cout << "LRP_MAXPOOLING2(): input dim: ";
    // input.showDim();
    // cout << "LRP_MAXPOOLING2(): output dim: ";
    // output.showDim();

    vector<int> inputDim = input.size();
    MatStruct outLRP(inputDim);
    int channelNum = inputDim[1];
    int dataDim = inputDim[0]; // Modify this to a vector if data is multi-dimensional.
    outLRP.zeroAssign();

    int channel_idx = 0;
    int data_idx = 0;
    for (channel_idx = 0; channel_idx < channelNum; channel_idx++){
        for (data_idx = 0; data_idx < dataDim; data_idx += 2){
            if (input.get(data_idx, channel_idx) < input.get(data_idx + 1, channel_idx)){
                outLRP.set(0.0, data_idx, channel_idx);
                outLRP.set(inLRP.get(data_idx/2, channel_idx), data_idx + 1, channel_idx);
            }else{
                outLRP.set(0.0, data_idx + 1, channel_idx);
                outLRP.set(inLRP.get(data_idx/2, channel_idx), data_idx, channel_idx);
            }
        }
    }

    return outLRP;
}

MatStruct LRP_FCBR(FCBRLayer* layerBody, MatStruct inLRP){
    cout << "Processing layer " << layerBody->id << endl;
    // Divied into 3 steps. inLRP -> ReLU, ReLU -> Bias, Bias -> FC.
    float zeroError = 0.00001; // Numbers less than zeroError are treated as zero.

    // inLRP -> ReLU
    MatStruct output = layerBody->relu->output->getMatStruct();

    // for debug
    // cout << "LRP_FCBR(): relu output dim: ";
    // output.showDim();

    inLRP.reshape(output.size());
    // set LRP zero if output of ReLU is zero
    long long int dataLength = output.getLength();
    long long int i = 0; // loop index
    for (i = 0; i < dataLength; i++){
        if (output.get(i) < zeroError){
            inLRP.set(0.0, i);
        }
    }

    // ReLU -> Bias
    output = layerBody->bias->output->getMatStruct(); 

    // for debug
    // cout << "LRP_FCBR(): bias output dim: ";
    // output.showDim();

    // Do nothing if bias is used to calculate LRP.
    // Otherwise, output of bias -= bias is needed to remove bias.
    
    
    // Bias -> FC
    MatStruct weight = layerBody->FC->weight->getMatStruct(); 
    MatStruct input = layerBody->FC->input->getMatStruct();
    vector<int> inputDim = input.size(); // Dimension of input should be a real number.
    vector<int> outputDim = output.size(); // Dimension of output should be a real number.
    int dataDim = 0; //inputDim[0];
    if (inputDim.size() > 1){
        dataDim = inputDim[0] * inputDim[1];
        inputDim.clear();
        inputDim.push_back(dataDim);
        input.reshape(inputDim);
    }else{
        dataDim = inputDim[0];
    }

    // for debug
    // cout << "LRP_FCBR(): FC input dim: ";
    // input.showDim();
    // cout << "LRP_FCBR(): FC weight dim: ";
    // weight.showDim();

    
    
    MatStruct outLRP(inputDim);
    outLRP.zeroAssign();

    int in_data_idx = 0;
    int out_data_idx = 0;
    /*
        i: in_data_idx
        j: out_data_idx
        x: input data
        y: output data
        w: weight
        Rx: outLRP
        Ry: inLRP
        Rx_i = x_i * sum_j Ry_j * w_ji/y_j , j ranges all out_data_idx
    */
    for (in_data_idx = 0; in_data_idx < dataDim; in_data_idx++){
        float temp = 0;
        for (out_data_idx = 0; out_data_idx < outputDim[0]; out_data_idx++){
            if (output.get(out_data_idx) > zeroError || output.get(out_data_idx) < -zeroError)
                temp += inLRP.get(out_data_idx) * weight.get(out_data_idx, in_data_idx)/output.get(out_data_idx);
        }
        temp *= input.get(in_data_idx);
        outLRP.set(temp, in_data_idx);
    }
    

    return outLRP;
}

MatStruct LRP_FCDO(FCDOLayer* layerBody, MatStruct inLRP){
    cout << "Processing layer " << layerBody->id << endl;
    float zeroError = 0.00001; // Numbers less than zeroError are treated as zero.

    MatStruct weight = layerBody->weight->getMatStruct(); 
    MatStruct input = layerBody->input->getMatStruct();
    MatStruct output = layerBody->output->getMatStruct();

    // for debug
    // cout << "LRP_FCDO(): FC input dim: ";
    // input.showDim();
    // cout << "LRP_FCDO(): FC weight dim: ";
    // weight.showDim();
    // cout << "LRP_FCDO(): FC output dim: ";
    // output.showDim();

    vector<int> inputDim = input.size(); // Dimension of input should be a real number.
    vector<int> outputDim = output.size(); // Dimension of output should be a real number.
    int dataDim = inputDim[0];
    MatStruct outLRP(inputDim);
    outLRP.zeroAssign();

    int in_data_idx = 0;
    int out_data_idx = 0;
    /*
        i: in_data_idx
        j: out_data_idx
        x: input data
        y: output data
        w: weight
        Rx: outLRP
        Ry: inLRP
        Rx_i = x_i * sum_j Ry_j * w_ji/y_j , j ranges all out_data_idx
    */
    for (in_data_idx = 0; in_data_idx < dataDim; in_data_idx++){
        float temp = 0;
        for (out_data_idx = 0; out_data_idx < outputDim[0]; out_data_idx++){
            if (output.get(out_data_idx) > zeroError || output.get(out_data_idx) < -zeroError)
                temp += inLRP.get(out_data_idx) * weight.get(out_data_idx, in_data_idx)/output.get(out_data_idx);
        }
        temp *= input.get(in_data_idx);
        outLRP.set(temp, in_data_idx);
    }
    

    return outLRP;
}


MatStruct doLRP(netStruct& myNet, int tag){
    MatStruct returnLRP;
    map<int, ProtoLayer*>::reverse_iterator rit;
    for(rit = myNet.net_inOrder.rbegin(); rit != myNet.net_inOrder.rend(); rit++){
        if(rit->second->layerType == LOGSOFTMAX){
            returnLRP = LRP_LOGSOFTMAX((LogSoftmaxLayer*) (rit->second), tag);
        }else if(rit->second->layerType == CBR1D){
            returnLRP = LRP_CBR1D((CBR1DLayer*) (rit->second), returnLRP);
        }else if(rit->second->layerType == MAXPOOLING2){
            returnLRP = LRP_MAXPOOLING2((MaxPooling2DLayer*) (rit->second), returnLRP);
        }else if(rit->second->layerType == FCBR){
            returnLRP = LRP_FCBR((FCBRLayer*) (rit->second), returnLRP);
        }else if(rit->second->layerType == FCDO){
            returnLRP = LRP_FCDO((FCDOLayer*) (rit->second), returnLRP);
        }else if(rit->second->layerType == NORMALIZE){
            // Do nothing.
        }
    }
    return returnLRP;
}

