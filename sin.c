#include <math.h>
#include <stdio.h>
#include <time.h>


#define N (int) 1e8
#define SIN sinf
#define FLOAT float

int main () {

    clock_t begin = clock();

    FLOAT sum = 0;

    #pragma acc parallel
    for (int i = 0; i < N; i++) {
        sum += SIN(i);
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin);

    printf("Time: %f\n", time_spent);
    printf("Sum: %f\n", sum);

    return 0;
}
