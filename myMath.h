#ifndef MYMATH
#define MYMATH



void permutation(int n, int m, int* result);

class randBag{
    private:
    int* permedIdx;
    int count;
    int partialSize;
    int allSize;

    public:
    randBag();
    randBag(int, int);
    ~randBag();
    void reset(int, int);
    int next();
    int soFar(); // This function returns so far where we are (index).
    randBag(const randBag& rhs);
    randBag& operator= (const randBag& rhs);
};

#endif
