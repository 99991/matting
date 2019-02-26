#include "common.h"
#include <math.h>

DLLEXPORT void label_expand(
    const double *image,
    const int *rows,
    const int *cols,
    int count,
    int r,
    int w,
    int h,
    const unsigned char *knownReg,
    double colorThresh,
    const double *trimap,
    double *extendedTrimap
){
    for (int k = 0; k < count; k++){
        int x = cols[k];
        int y = rows[k];
        
        double closest_dist = INFINITY;
        int closest_x = -1;
        int closest_y = -1;
        
        for (int dy = -r; dy <= r; dy++){
            for (int dx = -r; dx <= r; dx++){
                int x2 = x + dx;
                int y2 = y + dy;
                if (0 <= x2 && x2 < w && 0 <= y2 && y2 < h && knownReg[x2 + y2*w]){
                    double dr = image[(x + y*w)*3 + 0] - image[(x2 + y2*w)*3 + 0];
                    double dg = image[(x + y*w)*3 + 1] - image[(x2 + y2*w)*3 + 1];
                    double db = image[(x + y*w)*3 + 2] - image[(x2 + y2*w)*3 + 2];
                    
                    double color_distance = dr*dr + dg*dg + db*db;
                    
                    if (color_distance < colorThresh*colorThresh){
                        double dist = dx*dx + dy*dy;
                        
                        if (closest_dist > dist){
                            closest_dist = dist;
                            closest_x = x2;
                            closest_y = y2;
                        }
                    }
                }
            }
        }
        
        if (closest_x != -1){
            extendedTrimap[x + y*w] = trimap[closest_x + closest_y*w];
        }
    }
}
