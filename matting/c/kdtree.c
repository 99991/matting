#include "kdtree.h"

#include <stdlib.h>

static inline float kdtree_sq(float x){
    return x*x;
}

static inline float kdtree_clamp(float x, float a, float b){
    return x < a ? a : x > b ? b : x;
}

static float kdtree_point_distance(
    const float *a,
    const float *b,
    size_t dimension
){
    float sum = 0.0f;
    for (size_t i = 0; i < dimension; i++){
        sum += kdtree_sq(a[i] - b[i]);
    }
    return sum;
}

static size_t kdtree_partition(
    float *points,
    size_t *indices,
    size_t n,
    float pivot,
    size_t split_dim,
    size_t dimension
){
    size_t i = 0;
    size_t j = n - 1;
    
    while (i < j){
        while (i < j && points[i*dimension + split_dim] <  pivot) i++;
        while (i < j && points[j*dimension + split_dim] >= pivot) j--;

        // swap points
        if (i < j){
            float *a = points + i*dimension;
            float *b = points + j*dimension;
            
            for (size_t d = 0; d < dimension; d++){
                float temp = a[d];
                a[d] = b[d];
                b[d] = temp;
            }
            
            size_t temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
    }

    if (points[i*dimension + split_dim] < pivot) i++;
    
    return i;
}

struct kdtree* kdtree_init(
    float *points,
    size_t *indices,
    size_t n_points,
    size_t dimension
){
    if (n_points == 0) return NULL;

    struct kdtree *node = (struct kdtree*)malloc(sizeof(*node));

    float *lo = (float*)malloc(sizeof(*lo)*dimension);
    float *hi = (float*)malloc(sizeof(*hi)*dimension);

    node->lower_bounds = lo;
    node->upper_bounds = hi;

    for (size_t d = 0; d < dimension; d++){
        const float *p = points;
        
        lo[d] = p[d];
        hi[d] = p[d];
    }

    for (size_t i = 1; i < n_points; i++){
        const float *p = points + i*dimension;

        for (size_t d = 0; d < dimension; d++){
            if (lo[d] > p[d]) lo[d] = p[d];
            if (hi[d] < p[d]) hi[d] = p[d];
        }
    }
    
    node->dimension = dimension;
    
    if (n_points <= 20){
        node->less = NULL;
        node->more = NULL;
        node->split_dim = 0;
        node->split_value = 0.0f;
        node->points = points;
        node->indices = indices;
        node->n_points = n_points;
    }else{
        float max_length = 0.0f;
        size_t split_dim = 0;
        
        for (size_t d = 0; d < dimension; d++){
            float length = hi[d] - lo[d];
            
            if (max_length < length){
                max_length = length;
                split_dim = d;
            }
        }
        
        float split_value = lo[split_dim] + 0.5f*max_length;
        
        size_t n_less_points = kdtree_partition(
            points, indices, n_points, split_value, split_dim, dimension);
        
        // If all points are on one side, split in the middle.
        if (n_less_points == 0 || n_less_points == n_points){
            n_less_points = n_points/2;
        }
        
        size_t n_more_points = n_points - n_less_points;
        float *less_points = points;
        float *more_points = points + n_less_points*dimension;
        size_t *less_indices = indices;
        size_t *more_indices = indices + n_less_points;

        node->less = kdtree_init(less_points, less_indices, n_less_points, dimension);
        node->more = kdtree_init(more_points, more_indices, n_more_points, dimension);
        node->split_dim = split_dim;
        node->split_value = split_value;
        node->points = NULL;
        node->n_points = 0;
    }

    return node;
}

void kdtree_free(struct kdtree *node){
    if (!node) return;
    
    free(node->lower_bounds);
    free(node->upper_bounds);

    kdtree_free(node->less);
    kdtree_free(node->more);

    free(node);
}

static inline float kdtree_cell_distance(
    const float *p,
    const float *lo,
    const float *hi,
    size_t dimension
){
    float distance = 0.0f;
    for (size_t d = 0; d < dimension; d++){
        distance += kdtree_sq(p[d] - kdtree_clamp(p[d], lo[d], hi[d]));
    }
    return distance;
}

void kdtree_find_knn(
    struct kdtree *node,
    const float *query_point,
    struct kdtree_neighbor *neighbors,
    size_t *inout_n_neighbors,
    size_t k
){
    if (!node) return;

    size_t dimension = node->dimension;
    size_t n_neighbors = *inout_n_neighbors;

    float *lo = node->lower_bounds;
    float *hi = node->upper_bounds;
    
    if (n_neighbors >= k){
        float distance = neighbors[k - 1].distance;
        float cell_distance = kdtree_cell_distance(query_point, lo, hi, dimension);
        
        if (cell_distance > distance) return;
    }
    
    // if we found a leaf node, check its points
    if (node->points){
        for (size_t i_point = 0; i_point < node->n_points; i_point++){
            struct kdtree_neighbor *neighbor = &neighbors[n_neighbors++];
            neighbor->point = node->points + i_point*dimension;
            neighbor->index = node->indices[i_point];
            neighbor->distance = kdtree_point_distance(
                neighbor->point,
                query_point,
                dimension);

            // sort last neighbor into existing neighbors
            for (size_t j = n_neighbors - 1; j > 0; j--){
                if (neighbors[j - 1].distance > neighbors[j].distance){
                    struct kdtree_neighbor temp = neighbors[j];
                    neighbors[j] = neighbors[j - 1];
                    neighbors[j - 1] = temp;
                }else{
                    break;
                }
            }

            if (n_neighbors > k){
                n_neighbors = k;
            }
        }

        *inout_n_neighbors = n_neighbors;
    }else{
        // descend to child nodes
        if (query_point[node->split_dim] < node->split_value){
            kdtree_find_knn(node->less, query_point, neighbors, inout_n_neighbors, k);
            kdtree_find_knn(node->more, query_point, neighbors, inout_n_neighbors, k);
        }else{
            kdtree_find_knn(node->more, query_point, neighbors, inout_n_neighbors, k);
            kdtree_find_knn(node->less, query_point, neighbors, inout_n_neighbors, k);
        }
    }
}
