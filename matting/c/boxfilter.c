#include "common.h"
#include <assert.h>
#include <stdlib.h>

static inline double get_img(const double *img, int m, int n, int i, int j){
    return (0 <= i && i < m && 0 <= j && j < n) ? img[i*n + j] : 0.0;
}

static inline double get_row(const double *row, int n, int j){
    return (0 <= j && j < n) ? row[j] : 0.0;
}

DLLEXPORT void boxfilter(
    double *dst,
    int md,
    int nd,
    const double *src,
    int ms,
    int ns,
    int di,
    int dj,
    int r,
    int mode
){
    assert(mode == 0 || mode == 1 || mode == 2);
    
    assert(md >= 0);
    assert(nd >= 0);
    
    double *prev_row = (double*)malloc(ns*sizeof(*prev_row));
    
    int i = -1;
    for (int j = 0; j < ns; j++){
        double s = 0.0;
        for (int i2 = i - r + di; i2 < i + r + 1 + di; i2++){
            s += get_img(src, ms, ns, i2, j);
        }
        prev_row[j] = s;
    }
    
    for (int i = 0; i < md; i++){
        for (int j = 0; j < ns; j++){
            double s = prev_row[j];
            s -= get_img(src, ms, ns, i - r - 1 + di, j);
            s += get_img(src, ms, ns, i + r     + di, j);
            prev_row[j] = s;
        }
        
        int j = -1;
        double s = 0.0;
        for (int j2 = j - r + dj; j2 < j + r + 1 + dj; j2++){
            s += get_row(prev_row, ns, j2);
        }
        
        for (int j = 0; j < nd; j++){
            s -= get_row(prev_row, ns, j - r - 1 + dj);
            s += get_row(prev_row, ns, j + r     + dj);
            dst[i*nd + j] = s;
        }
    }

    free(prev_row);
}

#include <stdio.h>
#include <time.h>

static inline double sec(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

static void test(int r, int ms, int ns, int mode, int n_runs){
    int md, nd, di, dj;
    
    switch (mode){
    case 0:
        md = ms - 2*r;
        nd = ns - 2*r;
        di = r;
        dj = r;
        break;
    
    case 1:
        md = ms;
        nd = ns;
        di = 0;
        dj = 0;
        break;
    
    case 2:        
        md = ms + 2*r;
        nd = ns + 2*r;
        di = -r;
        dj = -r;
        break;
    }
    
    double *src = (double*)malloc(ms*ns*sizeof(*src));
    double *dst = (double*)malloc(md*nd*sizeof(*dst));
    
    for (int i = 0; i < ms*ns; i++){
        src[i] = rand() / (double)RAND_MAX;
    }
    
    for (int i = 0; i < md*nd; i++){
        dst[i] = rand() / (double)RAND_MAX;
    }
    
    for (int k = 0; k < n_runs; k++){
        
        double t = sec();
        
        boxfilter(dst, md, nd, src, ms, ns, di, dj, r, mode);
        
        double dt = sec() - t;
        
        printf("%f gbyte/sec\n", md*nd*8e-9/dt);
    }
    
    double max_error = 0.0;
    
    for (int i = 0; i < md; i++){
        for (int j = 0; j < nd; j++){
            double s = 0.0;
            for (int i2 = i - r + di; i2 <= i + r + di; i2++){
                for (int j2 = j - r + dj; j2 <= j + r + dj; j2++){
                    s += get_img(src, ms, ns, i2, j2);
                }
            }
            
            double error = s - dst[i*nd + j];
            
            if (error < 0.0) error = -error;
            
            if (max_error < error){
                max_error = error;
            }
        }
    }
    
    assert(max_error < 1e-10);
    
    free(src);
    free(dst);
}

int main(){
    test(4, 512, 512, 0, 10);
    
    for (int r = 1; r < 10; r++){
        int min_size = 2*r + 1;
        for (int ms = min_size; ms < min_size + 5*r; ms++){
            for (int ns = min_size; ns < min_size + 5*r; ns++){
                for (int mode = 0; mode <= 2; mode++){
                    test(r, ms, ns, mode, 1);
                }
            }
        }
    }
}
