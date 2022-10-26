#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT_TYPE double
#define MULT 128
#define N MULT * MULT

FLOAT_TYPE Anew[N][N];
FLOAT_TYPE A[N][N];

long iter_max = 10e6;
FLOAT_TYPE tol = 10e-6;

//nsys -t cuda,openacc -s none -w true ./main

int main()
{
    long iter = 0;
    FLOAT_TYPE err = 1.0;

    int pntA = 10, pntB = 20, pntC = 30, pntD = 20;

    #pragma acc kernels
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            A[j][i] = 0;
        }
    }

    A[0][0] = pntA;
    A[N - 1][0] = pntB;
    A[0][N - 1] = pntC;
    A[N - 1][N - 1] = pntD;
    

    for (int i = 1; i < N - 1; i++) A[i][0] = pntA + ((pntB - pntA) / N) * i;
    
    for (int i = 1; i < N - 1; i++) A[i][N - 1] = pntC + ((pntD - pntC) / N) * i;
    
    for (int i = 1; i < N - 1; i++) A[0][i] = pntA + ((pntC - pntA) / N) * i;
    
    for (int i = 1; i < N - 1; i++) A[N - 1][i] = pntB + ((pntD - pntB) / N) * i;

    // for (int i = 0; i < N; i++) {
    //     printf("\n");
    //     for (int j = 0; j < N; j++) {
    //         printf("%.0f ", A[j][i]);
    //     }
    // }

    #pragma acc data copy(A, Anew)
    while (err > tol && iter < iter_max)
    {
        err = 0.f;

        //#pragma omp parallel for shared(N, Anew, A)
        #pragma acc kernels
        for (int j = 1; j < N - 1; j++)
        {
            for (int i = 1; i < N - 1; i++)
            {
                Anew[j][i] = 0.25 * (A[j][i + 1] + A[j][i - 1] + A[j-1][i] + A[j + 1][i]);
                err = fmaxf(err, fabsf(Anew[j][i] - A[j][i]));
            }
        }

        //#pragma omp parallel for shared(N, Anew, A)
        #pragma acc kernels
        for( int j = 0; j < N - 1; j++) {
            for( int i = 0; i < N - 1; i++ ) {
                A[j][i] = Anew[j][i];
            }
        }

        if (iter % 100 == 0 || iter == 1) printf("%d, %0.6fn", iter, err);

        iter++;
        
    }

    // for (int i = 0; i < N; i++) {
    //     printf("\n");
    //     for (int j = 0; j < N; j++) {
    //         printf("%.0f ", A[j][i]);
    //     }
    // }

    getchar();
}
