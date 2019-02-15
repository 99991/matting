#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

void boxfilter_valid(
    const double *src,
    double *dst,
    int src_stride,
    int dst_stride,
    int nx,
    int ny,
    int r
){
    int nx2 = nx - 2*r;
    
    double *tmp = (double*)malloc(nx2*ny*sizeof(*tmp));
    
    for (int y = 0; y < ny; y++){
        double sum = 0.0;
        for (int x = 0; x <= 2*r; x++) sum += src[x + y*src_stride];
        tmp[0 + y*nx2] = sum;
        
        for (int x = r + 1; x < nx - r; x++){
            sum -= src[x - r - 1 + y*src_stride];
            sum += src[x + r     + y*src_stride];
            
            tmp[x - r + y*nx2] = sum;
        }
    }
    
    for (int x = 0; x < nx2; x++){
        double sum = 0.0;
        for (int y = 0; y <= 2*r; y++) sum += tmp[x + y*nx2];
        dst[x + 0*dst_stride] = sum;
        
        for (int y = r + 1; y < ny - r; y++){
            sum -= tmp[x + (y - r - 1)*nx2];
            sum += tmp[x + (y + r    )*nx2];
            
            dst[x + (y - r)*dst_stride] = sum;
        }
    }
    
    free(tmp);
}

// TODO inline this function into boxfilter_full
static inline double get(const double *data, int x, int y, int nx, int ny){
    return
        0 <= x &&
        x < nx &&
        0 <= y &&
        y < ny ?
        data[x + y*nx] : 0.0;
}

void boxfilter_full(
    const double *src,
    double *dst,
    int nx,
    int ny,
    int r
){
    int dst_stride = nx + 2*r;
    
    double *tmp = (double*)malloc((nx + 2*r)*(ny + 2*r)*sizeof(*tmp));

    // slow branchy 2*nx*ny
    for (int y = 0; y < ny + 2*r; y++){
        double sum = 0.0;
        for (int x = 0; x < nx + 2*r; x++){
            sum -= get(src, x - r - r - 1, y - r, nx, ny);
            sum += get(src, x, y - r, nx, ny);
            tmp[x + y*dst_stride] = sum;
        }
    }
    for (int x = 0; x < nx + 2*r; x++){
        double sum = 0.0;
        for (int y = 0; y < ny + 2*r; y++){
            sum -= get(tmp, x, y - r - 1, nx + 2*r, ny + 2*r);
            sum += get(tmp, x, y + r    , nx + 2*r, ny + 2*r);
            dst[x + y*dst_stride] = sum;
        }
    }
    
    free(tmp);
}
