#include "common.h"
#include <assert.h>
#include <stdlib.h>

// safely read image pixel
// if coordinates are not within image, return 0 instead
static inline double get_img(const double *img, int m, int n, int i, int j){
    return (0 <= i && i < m && 0 <= j && j < n) ? img[i*n + j] : 0.0;
}

// safely read row value
static inline double get_row(const double *row, int n, int j){
    return (0 <= j && j < n) ? row[j] : 0.0;
}

DLLEXPORT void boxfilter(
    double *dst,
    // height and width of dst
    int md,
    int nd,
    const double *src,
    // height and width of src
    int ms,
    int ns,
    // offset into src
    int di,
    int dj,
    // blur filter radius (blur kernel size will be 2r + 1)
    int r,
    // mode 0: valid (output -2r smaller)
    // mode 1: same  (output same size as input)
    // mode 2: full  (output 2r larger)
    int mode
){
    assert(r > 0);
    assert(mode == 0 || mode == 1 || mode == 2);
    assert(md >= 0);
    assert(nd >= 0);
    
    double *temp_row = (double*)malloc(ns*sizeof(*temp_row));
    
    // filter -1-th row vertically
    int i = -1;
    for (int j = 0; j < ns; j++){
        double s = 0.0;
        for (int i2 = i - r + di; i2 < i + r + 1 + di; i2++){
            s += get_img(src, ms, ns, i2, j);
        }
        temp_row[j] = s;
    }
    
    // innert part of row which does not need boundary checks
    int j0 = 0;
    // align inner region
    while ((j0 - r - 1 + dj) < 0 || (j0 - r - 1 - r) <= 0) j0++;
    
    for (int i = 0; i < md; i++){
        // filter first part of row vertically and then horizontally
        double prev_value = 0.0;
        int j2 = -1;
        for (int j = 0; j < j0; j++){
            // filter vertically
            double s = temp_row[j];
            s -= get_img(src, ms, ns, i - r - 1 + di, j);
            s += get_img(src, ms, ns, i + r     + di, j);
            temp_row[j] = s;
            
            // index of next row element that can be blurred horizontally
            j2 = j - r - dj;
            
            // if that element is within bounds
            if (0 <= j2 && j2 < nd){
                // if this is the first element of the row
                if (j2 == 0){
                    // filter the -1th-row value horizontally
                    for (int j2 = (-1) - r + dj; j2 < (-1) + r + 1 + dj; j2++){
                        prev_value += get_row(temp_row, ns, j2);
                    }
                }
                
                // filter horizontally
                prev_value -= get_row(temp_row, ns, j2 - r - 1 + dj);
                prev_value += get_row(temp_row, ns, j2 + r     + dj);
                dst[i*nd + j2] = prev_value;
            }
        }
        
        // filter inner part of row vertically and then horizontally
        for (int j = j0; j < ns; j++){
            double s = temp_row[j];
            // filter row value vertically
            // could make code three times longer to get rid of checks here
            // tried that, but no performance was gained
            s -= get_img(src, ms, ns, i - r - 1 + di, j);
            s += get_img(src, ms, ns, i + r     + di, j);
            temp_row[j] = s;
            
            // index of next row element that can be blurred horizontally
            j2 = j - r - dj;
            
            assert(0 <= j2 - r - 1 + dj);
            assert(j2 + r + dj < ns);
            
            // filter row value horizontally
            prev_value -= temp_row[j2 - r - 1 + dj];
            prev_value += temp_row[j2 + r     + dj];
            dst[i*nd + j2] = prev_value;
        }
        
        // filter remaining row horizontally
        assert(j2 > 0);
        j2++;
        for (int j = j2; j < nd; j++){
            prev_value -= get_row(temp_row, ns, j - r - 1 + dj);
            prev_value += get_row(temp_row, ns, j + r     + dj);
            dst[i*nd + j] = prev_value;
        }
    }
    
    free(temp_row);
}

#if 0
#include <stdio.h>
#include <time.h>

static inline double sec(){
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9*t.tv_nsec;
}

int compare_double(const void *p, const void *q){
    double a = *(double*)p;
    double b = *(double*)q;
    if (a < b) return -1;
    if (a > b) return +1;
    return 0;
}

static void test(int r, int ms, int ns, int mode, int n_runs){
    int md = -1, nd = -1, di = -1, dj = -1;
    
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
    
    if (md < 0 || nd < 0) return;
    
    double *src = (double*)malloc(ms*ns*sizeof(*src));
    double *dst = (double*)malloc(md*nd*sizeof(*dst));
    
    for (int i = 0; i < ms*ns; i++){
        src[i] = rand() / (double)RAND_MAX;
    }
    
    for (int i = 0; i < md*nd; i++){
        dst[i] = rand() / (double)RAND_MAX;
    }
    
    double *dts = (double*)malloc(n_runs*sizeof(*dts));
    
    for (int k = 0; k < n_runs; k++){
        
        double t = sec();
        
        boxfilter(dst, md, nd, src, ms, ns, di, dj, r, mode);
        
        double dt = sec() - t;
        
        dts[k] = dt;
        
        qsort(dts, k, sizeof(double), compare_double);
        
        double median_dt = dts[k / 2];
        
        printf("%f gbyte/sec - %f msec\n", md*nd*8e-9/median_dt, 1000*median_dt);
    }
    
    free(dts);
    
    double max_error = 0.0;
    
    if (md < 100 && nd < 100){
        
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
        
    }else{
        // sample random points
        for (int k = 0; k < 1000; k++){
            int i = rand() % md;
            int j = rand() % nd;
            
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
    for (int r = 1; r < 10; r++){
        for (int ms = 1; ms < 10; ms++){
            for (int ns = 1; ns < 10; ns++){
                for (int mode = 0; mode <= 2; mode++){
                    test(r, ms, ns, mode, 1);
                }
            }
        }
    }
    
    test(4, 123, 234, 0, 1);
    test(4, 456, 123, 0, 1);
    test(4, 13, 789, 0, 1);
    test(4, 567, 13, 0, 1);
    test(40, 1024, 1024, 0, 1);
    test(4, 512, 512, 0, 1);
    
    test(40, 1024, 1024, 0, 10000);
}
#endif
