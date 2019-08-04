#include "utils.h"
#include <stdlib.h>

void basic_sparsemm(const COO, const COO, COO *);
void basic_sparsemm_sum(const COO, const COO, const COO,
                        const COO, const COO, const COO,
                        COO *);

struct _csr_matrix {
    // A -> Stores all vals. Len NZ/
    double *A;
    // JA -> Stores cols of vals. Len NZ.
    int *JA;
    // Index of first item in a new row
    int *IA;
    // NZ
    int NZ;
};

typedef struct _csr_matrix *csr_matrix;

// Container for sorting items in B
struct container {
   int i, j;
   double data;
};

int convert_to_csr(const COO *matrix, csr_matrix *csr)
{
    // Alloc memory
    *csr = calloc(1, sizeof(struct _csr_matrix));

    (*csr)->A = (double *)malloc((*matrix)->NZ * sizeof(double));
    (*csr)->JA = (int *)malloc((*matrix)->NZ * sizeof(int));
    (*csr)->IA = (int *)malloc(((*matrix)->m+1) * sizeof(int));

    // For storing index of first NZ element in a row
    int counter = 0;
    // Keep track of last seen row
    int row = -1;
    // Keep track of total number of rows seen
    int total = 0;

    // Assign A and JA | Store vals in counter also
    // #pragma acc kernels
    for(int i=0; i < (*matrix)->NZ; i++){
        (*csr)->A[i] = (*matrix)->data[i];
        (*csr)->JA[i] = (*matrix)->coords[i].j;

        // IA tracks the index of the first item in any row
        if((*matrix)->coords[i].i != row){
            // Update prev seen row
            row = (*matrix)->coords[i].i;

            // Add index of first entry in new row to IA
            (*csr)->IA[counter] = i;
            counter++;

            // Increase total row count by one
            total++;
        }
    } 
    (*csr)->NZ = (*matrix)->NZ;

    // Add extra value to IA after most recent val (Buffer)
    (*csr)->IA[counter] = (*matrix)->NZ;

    // Return number of rows
    return total;
}

int convert_to_csc(const struct container *matrix, int NZ, int m, csr_matrix *csc){
    // Alloc memory
    *csc = calloc(1, sizeof(struct _csr_matrix));
    
    (*csc)->A = (double *)malloc(NZ * sizeof(double));
    (*csc)->JA = (int *)malloc(NZ * sizeof(int));
    (*csc)->IA = (int *)malloc((m+1) * sizeof(int));


    // # NZ elements in each column
    int counter = 0;
    // Keep track of last seen col
    int col = -1;
    // Total number of cols with vals in
    int total = 0;

    // Assign A and JA | Store vals in counter also
    // #pragma acc kernels
    for(int i=0; i < NZ; i++){
        (*csc)->A[i] = matrix[i].data;
        (*csc)->JA[i] = matrix[i].i;

        // IA tracks the index of the first item in any row
        if(matrix[i].j != col){
            // Update prev seen row
            col = matrix[i].j;

            // Add index of first entry in new row to IA
            (*csc)->IA[counter] = i;
            counter++;

            // Increase total columns by one
            total++;
        }
    } 
    (*csc)->NZ = NZ; 

    // Add extra value to IA after most recent val (Buffer)
    (*csc)->IA[counter] = NZ;

    // Return total number of cols
    return total;
}

void free_csr(csr_matrix *csr){
    free((*csr)->A);
    free((*csr)->JA);
    free((*csr)->IA);
    free((*csr));
}

// Comparitive Function for QSort
int cmpfunc (const void * a, const void * b) {
   int p = ((struct container *)a)->j;
   int q = ((struct container *)b)->j;
   return p-q;
}

/* Computes C = A*B.
 * C should be allocated by this routine.
 */
void optimised_sparsemm(const COO A, const COO B, COO *C)
{
    // --- Matrix size check --- 
    if (A->n != B->m) {
        fprintf(stderr, "Invalid matrix sizes, got %d x %d and %d x %d\n",
                A->m, A->n, B->m, B->n);
        exit(1);
    }

    // --- Preprep for Converting B to Csc ---
    // Sort B in Col, Row order (instead of initial Row, Col order)
    // Create struct list to store items in
    struct container *items;
    items = calloc(B->NZ, sizeof(struct container));

    // #pragma acc kernels
    for(int i=0; i < B->NZ; i++){
        // Create container for each entry
        struct container x;
        x.i = B->coords[i].i;
        x.j = B->coords[i].j;
        x.data = B->data[i];
        // Store entry in items
        items[i] = x;
    }

    // Sort items by j (col)
    qsort(items, B->NZ, sizeof(struct container), cmpfunc);

    // --- Convert A & B into Compressed Format ---
    csr_matrix X; // Pointer to struct obj
    csr_matrix Y; // Pointer to struct obj

    int row_total = convert_to_csr(&A, &X);  // Compressed Sparse Row -> row_total = # rows
    int col_total = convert_to_csc(items, B->NZ, B->m, &Y);  // Compressed Sparse Column -> col_total = # cols

    // --- Main Loop ---
    // Malloc Memory for current row
    double *d_row = (double*)malloc(A->n * sizeof(double));
    
    // Remove dependences on ->'s => Speed ++ & Parallelization ++
    int *X_IA, *X_JA;
    double *X_A;
    X_IA = X->IA;
    X_JA = X->JA;
    X_A = X->A;
    int X_n = A->n;
    
    int *Y_IA, *Y_JA;
    double *Y_A;
    Y_IA = Y->IA;
    Y_JA = Y->JA;
    Y_A = Y->A;
    int Y_n = B->n;

    // Create data var
    double data[A->m][B->n];
 
    // Loop through all rows (that contain data)
    for(int i = 0; i < row_total; i++){

        // Init row to 0
        for(int a = 0; a < A->n; a++){
            d_row[a] = 0;
        }

        // Convert current row of A to Dense
        int rowStart, rowEnd;
        rowStart = X_IA[i];
        rowEnd = X_IA[i+1];
        for(int j = rowStart; j < rowEnd; j++){  
            d_row[X_JA[j]] = X_A[j];
        }

        // Loop through all columns (that contain data)
        #pragma acc parallel loop
        for(int k=0; k < col_total; k++){
            // Init result
            double result = 0;
            // Loop through all items in that column
            int colStart, colEnd;
            colStart = Y_IA[k];
            colEnd = Y_IA[k+1];
            for(int p = colStart; p < colEnd; p++){
                result += Y_A[p] * d_row[Y_JA[p]];
            }

            data[i][k] = result;
            
        }

    }

    // Free row
    free(d_row);
    
    // Free other alloc'd memory
    free_csr(&X);
    free_csr(&Y);
    free(items);


    // For storing data in C
    double *_data = (double*)malloc(A->m*B->n*sizeof(double));
    struct coord *coords = (struct coord*)malloc(A->m*B->n*sizeof(struct coord));

    int i = 0;
    for(int r = 0; r < A->m; r++){
        for(int c = 0; c < B->n; c++){
            if(data[r][c] != 0){
                _data[i] = data[r][c];
                coords[i].i = r;
                coords[i].j = c;
                i++;
            }
        }
    }

    // Assign to C
    alloc_sparse(A->m, B->n, i, C);
    (*C)->data = _data;
    (*C)->coords = coords;

}

void add_matricies(const COO *A, const COO *B, COO *out){
    // Counters
    int x = 0, y = 0;

    // Loop until reach end of either A or B
    // #pragma acc kernels
    while(x < (*A)->NZ && y < (*B)->NZ){
        // If matching coors -> add vals
        if((*A)->coords[x].i == (*B)->coords[y].i && (*A)->coords[x].j == (*B)->coords[y].j){
            (*out)->coords[(*out)->NZ].i = (*A)->coords[x].i;
            (*out)->coords[(*out)->NZ].j = (*A)->coords[x].j;
            (*out)->data[(*out)->NZ] = (*A)->data[x] + (*B)->data[y];
            (*out)->NZ++;
            x++;
            y++;
        } 
        // If Row A < Row B
        else if ((*A)->coords[x].i < (*B)->coords[y].i) {
            (*out)->coords[(*out)->NZ].i = (*A)->coords[x].i;
            (*out)->coords[(*out)->NZ].j = (*A)->coords[x].j;
            (*out)->data[(*out)->NZ] = (*A)->data[x];
            (*out)->NZ++;
            x++;
        // If Row B < Row A
        } else if ((*B)->coords[y].i < (*A)->coords[x].i ){
            (*out)->coords[(*out)->NZ].i = (*B)->coords[y].i;
            (*out)->coords[(*out)->NZ].j = (*B)->coords[y].j;
            (*out)->data[(*out)->NZ] = (*B)->data[y];
            (*out)->NZ++;
            y++;
        // If Row A = Row B
        } else {
            // If Col A < Col B
            if((*A)->coords[x].j < (*B)->coords[y].j){
                (*out)->coords[(*out)->NZ].i = (*A)->coords[x].i;
                (*out)->coords[(*out)->NZ].j = (*A)->coords[x].j;
                (*out)->data[(*out)->NZ] = (*A)->data[x];
                (*out)->NZ++;
                x++;
            // If Col B < Col A
            } else {
                (*out)->coords[(*out)->NZ].i = (*B)->coords[y].i;
                (*out)->coords[(*out)->NZ].j = (*B)->coords[y].j;
                (*out)->data[(*out)->NZ] = (*B)->data[y];
                (*out)->NZ++;
                y++;
            }
        }
    }

    // When here, will have finished through one of the arrays
    // If @ End of A -> Finish B
    if(x == (*A)->NZ){
        // #pragma acc kernels
        while(y < (*B)->NZ){
            (*out)->coords[(*out)->NZ].i = (*B)->coords[y].i;
            (*out)->coords[(*out)->NZ].j = (*B)->coords[y].j;
            (*out)->data[(*out)->NZ] = (*B)->data[y];
            (*out)->NZ++; 
            y++;
        }
    // If @ End of B -> Finish A
    } else {
        // #pragma acc kernels
        while(x < (*A)->NZ){
            (*out)->coords[(*out)->NZ].i = (*A)->coords[x].i;
            (*out)->coords[(*out)->NZ].j = (*A)->coords[x].j;
            (*out)->data[(*out)->NZ] = (*A)->data[x];
            (*out)->NZ++;
            x++;
        }
    }
    
}


/* Computes O = (A + B + C) (D + E + F).
 * O should be allocated by this routine.
 */
void optimised_sparsemm_sum(const COO A, const COO B, const COO C,
                            const COO D, const COO E, const COO F,
                            COO *O)
{
    COO X, Y, P, Q;
    // Alloc intermediates
    alloc_sparse(A->m, A->n, A->m*A->n, &X); // ABC same size
    alloc_sparse(A->m, A->n, A->m*A->n, &Y); // ABC same size
    alloc_sparse(D->m, D->n, D->m*D->n, &P); // DEF same size
    alloc_sparse(D->m, D->n, D->m*D->n, &Q); // DEF same size

    // Set NZ back to 0 for intermediates
    X->NZ = 0;
    Y->NZ = 0;
    P->NZ = 0;
    Q->NZ = 0;

    // A + B = X
    add_matricies(&A, &B, &X);
    // (A + B) = X + C = Y
    add_matricies(&X, &C, &Y);
    // D + E = P
    add_matricies(&D, &E, &P);
    // (D + E) = P + F = Q
    add_matricies(&P, &F, &Q);

    // Y * Q - use mult routine from above
    optimised_sparsemm(Y, Q, O);

    // Free intermediates
    free_sparse(&X);
    free_sparse(&Y);
    free_sparse(&P);
    free_sparse(&Q);

}
