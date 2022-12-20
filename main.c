#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FLOAT_TYPE double
#define N 256

struct timeval tv1, tv2, dtv;
struct timezone *tz;

void time_start() { gettimeofday(&tv1, &tz); }

long time_stop()
{ 
    gettimeofday(&tv2, &tz);
    dtv.tv_sec= tv2.tv_sec -tv1.tv_sec;
    dtv.tv_usec=tv2.tv_usec-tv1.tv_usec;
    if(dtv.tv_usec<0) { dtv.tv_sec--; dtv.tv_usec+=1000000; }
    return dtv.tv_sec*1000+dtv.tv_usec/1000;
}

long iter_max = 10e5;
FLOAT_TYPE tol = 10e-6;

//pgcc -acc -Minfo=accel -o main main.c -lm && PGI_ACC_TIME=1

//nsys -t cuda,openacc -s none -w true ./main

FLOAT_TYPE **allocate2DArray(int row, int col)
{
    FLOAT_TYPE ** ptr = (FLOAT_TYPE **) malloc(sizeof(FLOAT_TYPE *)*row);
    for(int i = 0; i < row; i++)
        ptr[i] = (FLOAT_TYPE *) malloc(sizeof(FLOAT_TYPE)*col);

    return ptr;
}

FLOAT_TYPE diff(int nx, int ny, FLOAT_TYPE **a, FLOAT_TYPE **b) 
{
    FLOAT_TYPE v = 0.0, t;
    #pragma acc kernels loop reduction(+:v)
    for (int j = 0; j < ny; j++) 
        for (int i = 0; i < nx; i++)
            t = a[i][j] - b[i][j];  v += t * t;

    return sqrt(v / (FLOAT_TYPE)(nx * ny));
}

void init_border(uint n, FLOAT_TYPE **T, FLOAT_TYPE left_top, FLOAT_TYPE right_top, FLOAT_TYPE left_bottom, FLOAT_TYPE right_bottom)
{
    for (int j = 1; j < n - 1; j++) 
      for (int i = 1; i < n - 1; i++) 
        T[i][j] = 0.0;

    for(uint i = 0; i < n; i++){
        T[i][0] = left_top + i*(right_top - left_top) / (n - 1);
        T[i][n - 1] = left_bottom + i*(right_bottom - left_bottom) / (n - 1);

        T[0][i] = left_bottom + i*(left_bottom - left_top) / (n - 1);
        T[n - 1][i] = right_top + i*(right_bottom - right_top) / (n - 1);
    }
}

int main()
{
    iter_max--;
    FLOAT_TYPE **A = allocate2DArray(N, N);
    FLOAT_TYPE **Anew = allocate2DArray(N, N);

    init_border(N, A, 10, 20, 20, 30);

    long iter = 0;
    FLOAT_TYPE err = 10000.;

    time_start();

    #pragma acc data copy(A[:N][:N]) create(Anew[:N][:N]) create(err)
    while (err >= tol && iter <= iter_max)
    {
        #pragma acc parallel loop independent
        for (int j = 1; j < N - 1; j++)
            for (int i = 1; i < N - 1; i++)
                Anew[i][j] = 0.25 * (
                    A[i - 1][j] + A[i][j + 1] +
                    A[i][j - 1] + A[i + 1][j]
                    );

        err = diff(N, N, Anew, A);

        #pragma acc parallel loop independent
        for( int j = 0; j < N - 1; j++)
            for( int i = 0; i < N - 1; i++ )
                A[i][j] = Anew[i][j];

        if (iter % 100 == 0 || iter == 0) printf("%d, %0.13fn\n", iter, err);

        iter++;
        
    }

    printf("Iterations: %d\n", iter);
    printf("Err: %f\n", err);

    printf("Time: %ld\n", time_stop());

    return 0;
}
