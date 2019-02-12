#ifndef KDTREE_INCLUDED
#define KDTREE_INCLUDED

#include <stddef.h>

struct kdtree_neighbor {
    float *point;
    size_t index;
    float distance;
};

// kd-tree node
struct kdtree {
    // node data (only exists if node is a leaf)
    float *points;
    size_t *indices;
    size_t n_points;
    // bounding box of points
    float *lower_bounds;
    float *upper_bounds;
    
    // children
    struct kdtree *less;
    struct kdtree *more;
    // split plane
    size_t split_dim;
    float split_value;
    
    size_t dimension;
};

struct kdtree* kdtree_init(
    float *points,
    size_t *indices,
    size_t n_points,
    size_t dimension
);

void kdtree_free(struct kdtree *node);

void kdtree_find_knn(
    struct kdtree *tree,
    // point to search for
    const float *query_point,
    // neighbors to find
    struct kdtree_neighbor *neighbors,
    size_t *inout_n_neighbors,
    // number of neighbors to find
    size_t k
);

#endif
