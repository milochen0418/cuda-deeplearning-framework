#ifndef TAGTOOL
#define TAGTOOL

//#include "kernel.h"
#include <map>
#include <vector>
#include <string>
#include "matrixtool3.h"
#include "myMath.h"

using namespace std;


class tagAddress{
// One tagAddress means one address of some tag of some brain, meaning:
// the tag of voxel this->tagIdx of brain this->sampleIdx is this->tag.
    public:
    int sampleIdx; // sampleIdx records the index of the brain.
    int tag; // tag is the tag
    vector<int> tagIdx; // tagIdx is the index of this voxel in brain sampleIdx.
};


class tagBag{
// loadTag (or use constructor) -> setBZone (or makeTag) -> setTag (if need) -> getAPixel

    private:
    MatIntStruct rawTag, rawTagBackup;
    map<int, vector<tagAddress> > tagBundle; // This variable stores the types of tags.
    map<int, randBag> ballBag;
    

    public:
    //MatIntStruct rawTag, rawTagBackup;
    tagBag();
    tagBag(string fileName);
    void makeTag();
    void shuffle(int tagType);
    void shuffleAll();
    void loadTag(string fileName);
    void setTag(int tagType, int tagYouWant);// This function only changes the tag in tagAddress, not the original tag.
    void setBZone(int); // This set a boundary zone to the tag, so that voxel in the zone will never be chosen.
    tagAddress getAPixel(int tagType);// Use the original tagType, not the tagYouWant used in setTag()

    void showTagType();
    vector<int> getTagType();
    int getTagLength(int tagType);
};


class dataPackTool{
    private:
    MatStruct allData;

    public:
    dataPackTool();
    dataPackTool(string fileName);
    void loadDataPack(string fileName);
    MatStruct extractBlock(int sampleIdx, vector<int> centerCoor, vector<int> blockWidth);
    MatStruct extractBlock(tagAddress, vector<int> blockWidth);
};



#endif
