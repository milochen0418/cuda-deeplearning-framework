#include "tagtool.h"
#include <map>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>

using namespace std;

tagBag::tagBag(string fileName){
    loadTag(fileName);
}

void tagBag::setTag(int tagType, int tagYouWant){
    for(int i=0; i<tagBundle[tagType].size(); i++){
        tagBundle[tagType][i].tag = tagYouWant;
    }
}


void tagBag::loadTag(string fileName){
    rawTag = MatIntFromFile(fileName);
    rawTagBackup = rawTag;
    cout << "tagBag::loadTag() rawTag dimension: ";
    rawTag.showDim();
}

void tagBag::makeTag(){
    int i, j;
    bool exist;
    vector<int> tagType;
    tagBundle.clear();

    // Check tag types.
    cout << "tagBag::makeTag() Searching for tag types...\n";
    for(i=0; i<rawTag.getLength(); i++){
        exist = false;
        for(j=0; j<tagType.size(); j++){
            if(((int) rawTag.get(i)) == tagType[j]){
                exist = true;
                break;
            }
        }
        if(!exist){
            cout << "tagBag::makeTag() new tag type: " << (int) rawTag.get(i) << endl;
            tagType.push_back((int) rawTag.get(i));
        }
    }

    cout << "tagBag::makeTag() tag types number: " << tagType.size() << endl;

    for(i=0; i<tagType.size(); i++){
        vector<tagAddress> temp;
        tagBundle[tagType[i]] = temp;
    }

    cout << "tagBag::makeTag() Arranging tags...\n";

    for(i=0; i<rawTag.getLength(); i++){
        tagAddress temp;
        vector<int> coor = rawTag.linear2multiD(i);
        temp.sampleIdx = coor[coor.size()-1];
        temp.tag = (int) rawTag.get(i);
        for(j=0; j<coor.size()-1; j++){
            temp.tagIdx.push_back(coor[j]);
        }
        tagBundle[(int) rawTag.get(i)].push_back(temp);
    }

    ballBag.clear();

    for(i=0; i<tagType.size(); i++){
        randBag temp;
        ballBag[tagType[i]] = temp;
    }

    cout << "tagBag::makeTag() Shuffling tags...\n";

    shuffleAll();
    cout << "tagBag::makeTag() End of makeTag.\n\n";
}

void tagBag::shuffle(int tagType){
    int tagNum = getTagLength(tagType);
    //cout << "tagBag::shuffle"
    ballBag[tagType].reset(tagBundle[tagType].size(), tagBundle[tagType].size());
}

void tagBag::shuffleAll(){
    map<int, vector<tagAddress> >::iterator it;
    for(it = tagBundle.begin(); it != tagBundle.end(); it++){
        cout << "tagBag::shuffleAll() Shuffling tag type: " << it->first << ", tag number: " << it->second.size() << endl;
        shuffle(it->first);
    }
}

int tagBag::getTagLength(int tagType){
    map<int, vector<tagAddress> >::iterator it;
    it = tagBundle.find(tagType);
    if(it == tagBundle.end()){
        cout << "tagBag::getTagLength() No tagType " << tagType << ".\n";
        return(-1);
    }else{
        return(tagBundle[tagType].size());
    }
}


void tagBag::setBZone(int boundaryWidth){
    //cout << "tagBag::setBZone() here 1\n";
    vector<int> coor1, coor2;
    vector<int> dimension = rawTag.size();
    int i, j;
    vector<vector<int> > oldBZone = rawTag.find("==", -1.0);
    
    for(i=0; i<oldBZone.size(); i++){
        rawTag.set(rawTagBackup.get(oldBZone[i]), oldBZone[i]);
    }

    //cout << "tagBag::setBZone() here 2\n";
    
    for(i=0; i<dimension.size()-1; i++){
        coor1.clear();
        coor2.clear();
        coor1.assign(dimension.size(), -1);
        coor2.assign(dimension.size(), -1);

        coor1[i] = 0;
        coor2[i] = boundaryWidth - 1;
        rawTag.setCertain(-1, coor1, coor2);
    }

    //cout << "tagBag::setBZone() here 3\n";

    for(i=0; i<dimension.size()-1; i++){
        coor1.clear();
        coor2.clear();
        coor1.assign(dimension.size(), -1);
        coor2.assign(dimension.size(), -1);

        coor1[i] = dimension[i] - boundaryWidth;
        coor2[i] = dimension[i] - 1;
        rawTag.setCertain(-1, coor1, coor2);
    }

    //cout << "tagBag::setBZone() here 4\n";
    
    makeTag();
}

tagAddress tagBag::getAPixel(int tagType){
    map<int, vector<tagAddress> >::iterator it;
    it = tagBundle.find(tagType);
    
    if(it == tagBundle.end()){
        cout << "tagBag::getAPixel() No such tagType.\n";
        tagAddress badans;
        return(badans);
    }else{
        return(tagBundle[tagType][ballBag[tagType].next()]);
    }
}

void tagBag::showTagType(){
    map<int, vector<tagAddress> >::iterator it;
    cout << "tag type: ";

    for(it = tagBundle.begin(); it != tagBundle.end(); it++){
        cout << it->first << " ";
    }
    cout << endl;
}

vector<int> tagBag::getTagType(){
    vector<int> ans;
    map<int, vector<tagAddress> >::iterator it;
    for(it = tagBundle.begin(); it != tagBundle.end(); it++){
        ans.push_back(it->first);
    }
    return(ans);
}

tagBag::tagBag(){}

//============================================================

dataPackTool::dataPackTool(string fileName){
    loadDataPack(fileName);
}

dataPackTool::dataPackTool(){}

void dataPackTool::loadDataPack(string fileName){
    allData = MatFromFile(fileName);
    cout << "dataPackTool::loadDataPack() allData dimension: ";
    allData.showDim();
}


MatStruct dataPackTool::extractBlock(int sampleIdx, vector<int> centerCoor, vector<int> blockWidth){
    int i, left, right;
    bool even, busted;
    MatStruct ans;
    int halfWidth;
    vector<int> dataDim = allData.size();
    vector<int> coor1, coor2;

    if(centerCoor.size()+2 != dataDim.size()){
        cout << "dataPackTool::extractBlock() centerCoor may not be consistent with the dimension of allData.\n";
    }

    busted = false;
    for(i=0; i<blockWidth.size(); i++){
        even = (blockWidth[i] % 2 == 0);
        if(even){
            halfWidth = blockWidth[i]/2;
            left = centerCoor[i] - halfWidth + 1;
            right = centerCoor[i] + halfWidth;
        }else{
            halfWidth = (blockWidth[i]-1)/2;
            left = centerCoor[i] - halfWidth;
            right = centerCoor[i] + halfWidth;
        }

        coor1.push_back(left);
        coor2.push_back(right);
        // check busted
        if((left < 0) || (right >= dataDim[i])){
            cout << "dataPackTool::extractBlock() blockWidth out of range.\n";
            busted = true;
            break;
        } 
    }
    if(!busted){
        coor1.push_back(-1);
        coor2.push_back(-1);
        coor1.push_back(sampleIdx);
        coor2.push_back(sampleIdx);
        ans = allData.get(coor1, coor2);
    }

    return(ans);
}

MatStruct dataPackTool::extractBlock(tagAddress sampleTag, vector<int> blockWidth){
    int i, left, right;
    bool even, busted;
    MatStruct ans;
    int halfWidth;
    vector<int> dataDim = allData.size();
    vector<int> coor1, coor2;
    int sampleIdx = sampleTag.sampleIdx;
    vector<int> centerCoor = sampleTag.tagIdx;

    if(centerCoor.size()+2 != dataDim.size()){
        cout << "dataPackTool::extractBlock() centerCoor may not be consistent with the dimension of allData.\n";
    }

    busted = false;
    for(i=0; i<blockWidth.size(); i++){
        even = (blockWidth[i] % 2 == 0);
        if(even){
            halfWidth = blockWidth[i]/2;
            left = centerCoor[i] - halfWidth + 1;
            right = centerCoor[i] + halfWidth;
        }else{
            halfWidth = (blockWidth[i]-1)/2;
            left = centerCoor[i] - halfWidth;
            right = centerCoor[i] + halfWidth;
        }

        coor1.push_back(left);
        coor2.push_back(right);
        // check busted
        if((left < 0) || (right >= dataDim[i])){
            cout << "dataPackTool::extractBlock() blockWidth out of range.\n";
            busted = true;
            break;
        } 
    }
    if(!busted){
        coor1.push_back(-1);
        coor2.push_back(-1);
        coor1.push_back(sampleIdx);
        coor2.push_back(sampleIdx);
        ans = allData.get(coor1, coor2);
    }

    return(ans);
}