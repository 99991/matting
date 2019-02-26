#include "common.h"
#include "kdtree.h"
#include <stdlib.h>
#include <assert.h>

// Using int type for sizes instead of size_t due to 32/64-bit issues.
DLLEXPORT void knn(
    float *data_points,
    float *query_points,
    int *out_neighbor_indices,
    float *out_neighbor_squared_distances,
    const int n_data_points,
    const int n_query_points,
    const int point_dimension,
    const int k
){
    // Ensure that function is not used incorrectly.
    // Can't query for k data points if there are less than k.
    assert(k <= n_data_points);
    if (k > n_data_points) exit(-1);
    
    size_t *indices = (size_t*)malloc(n_data_points*sizeof(*indices));
    
    for (int i = 0; i < n_data_points; i++){
        indices[i] = i;
    }
    
    struct kdtree *tree = kdtree_init(
        data_points,
        indices,
        n_data_points,
        point_dimension);

    // Working array size must be 1 greater then k
    struct kdtree_neighbor *neighbors = (struct kdtree_neighbor*)malloc((k + 1)*sizeof(*neighbors));
    
    for (int i = 0; i < n_query_points; i++){
        float *query_point = query_points + i*point_dimension;

        size_t n_neighbors = 0;
        kdtree_find_knn(tree, query_point, neighbors, &n_neighbors, k);

        for (size_t j = 0; j < n_neighbors; j++){
            *out_neighbor_indices++ = neighbors[j].index;
            *out_neighbor_squared_distances++ = neighbors[j].distance;
        }
    }

    free(neighbors);
    kdtree_free(tree);
    free(indices);
}
