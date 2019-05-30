#include "common.h"
#include <math.h>
#include <stdlib.h>

int compare_int(const void *p, const void *q){
    int a = *(int*)p;
    int b = *(int*)q;
    
    return
        a < b ? -1 :
        a > b ? +1 :
        0;
}

DLLEXPORT void ichol_free(void *ptr){
    free(ptr);
}

#define ICHOL_NOT_POSITIVE_DEFINITE -1
#define ICHOL_OUT_OF_MEMORY -2

DLLEXPORT int ichol(
    const double *A_data,
    const int *A_indices,
    const int *A_indptr,
    double **L_data_out,
    int **L_indices_out,
    int **L_indptr_out,
    int n,
    double threshold,
    int discard_if_zero_in_A
){
    int     L_nnz       = 0;
    int     L_max_nnz   = 0;
    double *L_data      = NULL;
    int    *L_indices   = NULL;
    int    *L_indptr    = (int*)malloc((n+1)*sizeof(*L_indptr));
    int    *L_starts    = (int*)malloc(n*sizeof(*L_starts));
    int    *L_row_lists = (int*)malloc(n*sizeof(*L_row_lists));
    
    int     col_j_nnz     = 0;
    int    *col_j_indices = (int*)malloc(n*sizeof(*col_j_indices));
    int    *col_j_used    = (int*)malloc(n*sizeof(*col_j_used));
    double *col_j_data    = (double*)malloc(n*sizeof(*col_j_data));
    
    int error = 0;
    
    // Check if malloc failed
    if (
        !L_indptr ||
        !L_starts ||
        !L_row_lists ||
        !col_j_indices ||
        !col_j_used ||
        !col_j_data
    ){
        error = ICHOL_OUT_OF_MEMORY;
        goto cleanup;
    }
    
    // Initialization
    for (int i = 0; i < n; i++){
        L_row_lists[i] = -1;
        col_j_used[i] = 0;
        col_j_data[i] = 0.0;
        
        L_indptr[i] = 0;
        L_starts[i] = 0;
        col_j_indices[i] = 0;
    }
    
    // For each column j
    for (int j = 0; j < n; j++){
        
        // Load column j from lower triangular part of A
        for (int idx = A_indptr[j]; idx < A_indptr[j+1]; idx++){
            int i = A_indices[idx];
            // Only load lower-triangular part, including diagonal
            if (i >= j){
                double A_ij = A_data[idx];
                
                col_j_data[i] = A_ij;
                col_j_used[i] = 1;
                col_j_indices[col_j_nnz] = i;
                col_j_nnz++;
            }
        }
        
        // For each non-zero element in row k
        int k = L_row_lists[j];
        while (k != -1){
            int next_k = L_row_lists[k];
            int col_k_start = L_starts[k];
            int col_k_end = L_indptr[k + 1];
            double L_jk = L_data[col_k_start];
            
            // For each non-zero element in column k, starting at j
            for (int col_k_idx = col_k_start; col_k_idx < col_k_end; col_k_idx++){
                if (discard_if_zero_in_A){
                    int i = L_indices[col_k_idx];
                    
                    // Discard update on L_ij if using sparsity pattern of A
                    if (!col_j_used[i]) continue;
                    
                    double L_ik = L_data[col_k_idx];
                    col_j_data[i] -= L_ik*L_jk;
                }else{
                    int i = L_indices[col_k_idx];
                    double L_ik = L_data[col_k_idx];
                    col_j_data[i] -= L_ik*L_jk;
                    
                    // If row i in column j does not exist yet, create it
                    if (!col_j_used[i]){
                        col_j_used[i] = 1;
                        col_j_indices[col_j_nnz] = i;
                        col_j_nnz++;
                    }
                }
            }
            
            if (col_k_start + 1 < col_k_end){
                col_k_start++;
                int i = L_indices[col_k_start];
                L_row_lists[k] = L_row_lists[i];
                L_row_lists[i] = k;
                L_starts[k] = col_k_start;
            }
            
            k = next_k;
        }
        
        // Check if matrix is not positive definite
        if (!col_j_used[j] || col_j_data[j] <= 0.0){
            error = ICHOL_NOT_POSITIVE_DEFINITE;
            goto cleanup;
        }
        
        col_j_data[j] = sqrt(col_j_data[j]);

        // Divide column by diagonal element
        for (int idx = 1; idx < col_j_nnz; idx++){
            int i = col_j_indices[idx];
            
            col_j_data[i] /= col_j_data[j];
        }
        
        // Allocate memory for new elements from column j
        if (L_nnz + col_j_nnz > L_max_nnz){
            // New memory size = 3/2 * old memory size + column size
            L_max_nnz = L_max_nnz + (L_max_nnz >> 1) + col_j_nnz;
            
            int *L_indices_new = (int*)realloc(L_indices, L_max_nnz*sizeof(*L_indices));
            double *L_data_new = (double*)realloc(L_data, L_max_nnz*sizeof(*L_data));
            
            // Check if allocation failed
            if (!L_indices_new || !L_data_new){
                free(L_indices_new);
                free(L_data_new);
                
                error = ICHOL_OUT_OF_MEMORY;
                goto cleanup;
            }else{
                L_indices = L_indices_new;
                L_data = L_data_new;
            }
        }
        
        L_starts[j] = L_nnz + 1;

        // Sort column j by row index i
        qsort(col_j_indices, col_j_nnz, sizeof(int), compare_int);
        
        for (int idx = 0; idx < col_j_nnz; idx++){
            int i = col_j_indices[idx];
            
            double L_ij = col_j_data[i];
            col_j_used[i] = 0;
            col_j_data[i] = 0.0;
            
            if (fabs(L_ij) >= threshold){
                L_indices[L_nnz] = i;
                L_data[L_nnz] = L_ij;
                L_nnz++;
            }
        }
        
        L_indptr[j + 1] = L_nnz;
        col_j_nnz = 0;
        
        if (L_indptr[j] + 1 < L_indptr[j + 1]){
            int i = L_indices[L_indptr[j] + 1];
            L_row_lists[j] = L_row_lists[i];
            L_row_lists[i] = j;
        }
    }
    
cleanup:
    if (error == 0){
        *L_data_out = L_data;
        *L_indices_out = L_indices;
        *L_indptr_out = L_indptr;
    }else{
        free(L_data);
        free(L_indices);
        free(L_indptr);
        
        *L_data_out = NULL;
        *L_indices_out = NULL;
        *L_indptr_out = NULL;
    }
    
    free(L_starts);
    free(L_row_lists);
    
    free(col_j_indices);
    free(col_j_used);
    free(col_j_data);
    
    return error;
}

DLLEXPORT void backsub_L_csc_inplace(
    const double *L_data,
    const int *L_indices,
    const int *L_indptr,
    double *x,
    int n
){
    for (int j = 0; j < n; j++){
        int k = L_indptr[j];
        double L_jj = L_data[k];
        double temp = x[j]/L_jj;
        
        x[j] = temp;
        
        for (int k = L_indptr[j] + 1; k < L_indptr[j + 1]; k++){
            int i = L_indices[k];
            double L_ij = L_data[k];
            
            x[i] -= L_ij*temp;
        }
    }
}

DLLEXPORT void backsub_LT_csc_inplace(
    const double *L_data,
    const int *L_indices,
    const int *L_indptr,
    double *x,
    int n
){
    for (int i = n - 1; i >= 0; i--){
        double s = x[i];
        
        for (int k = L_indptr[i] + 1; k < L_indptr[i + 1]; k++){
            int j = L_indices[k];
            double L_ji = L_data[k];
            s -= L_ji*x[j];
        }
        
        int k = L_indptr[i];
        double L_ii = L_data[k];
        
        x[i] = s/L_ii;
    }
}

DLLEXPORT void backsub_L_csr_inplace(
    const double *L_data,
    const int *L_indices,
    const int *L_indptr,
    double *x,
    int n
){
    for (int i = 0; i < n; i++){
        double s = x[i];
        
        int k_end = L_indptr[i+1] - 1;
        for (int k = L_indptr[i]; k < k_end; k++){
            int j = L_indices[k];
            double L_ij = L_data[k];
            s -= L_ij*x[j];
        }
        
        int k = k_end;
        double L_ii = L_data[k];
        
        x[i] = s/L_ii;
    }
}
