#include <cstdlib>
#include <cstdio>
#include <cuda.h>

using namespace std;

__global__ void mykernel(void) {
}

int main(void) {
mykernel<<<1,1>>>();
printf("CPU Hello World!\n");
return 0;
} 
