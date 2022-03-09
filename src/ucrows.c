/* Communication-efficient distributed stratified stochastic gradient decent 
 * Copyright © 2022 Nabil Abubaker (abubaker.nf@gmail.com)
 * 
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 * OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "basic.h"
#include "comm.h"
#include "def.h"
#include "util.h"
#include "ucrows.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#define CHECK_BIT(var, pos) ((var[pos/32]) & (1<<(pos%32)))

static inline void setBit(int *arr, idx_t bit_idx) {
    arr[bit_idx / 32] |= 1 << (bit_idx % 32);
}

static inline void unsetBit(int *arr, idx_t bit_idx) {
    arr[bit_idx / 32] &= ~(1 << (bit_idx % 32));
}

void get_shared_cols_2(triplet *M, idx_t nnz, idx_t no_global_cols, int *col_update_order, int *order); 
/* this function prepares the necessary columns info for each processor  */
/* void get_shared_cols_2(triplet *M, idx_t nnz, idx_t no_global_cols, int *col_update_order, int *order) {
 *     int sz = sizeof(int) * 8, nprocs, mypid ;
 *     idx_t no_bs = (no_global_cols / sz) + 1, i, j;
 *     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
 *     MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
 *     int *mybitstring = calloc(no_bs * nprocs, sizeof(*mybitstring));
 *     for (i = 0; i < nnz; ++i) {
 *         assert(M[i].col < no_global_cols);
 *         setBit(mybitstring + (no_bs * mypid), M[i].col);
 *     }
 * 
 * #ifdef UGLY_DBG
 *     if (mypid == 0) {
 *         printf("Before Comm:\n");
 *         for (j = 0; j < no_bs; ++j) {
 *             printf("" PRINTF_BINARY_PATTERN_INT32 " ",
 *                     PRINTF_BYTE_TO_BINARY_INT32(mybitstring[mypid * no_bs + j]));
 *         }
 *         printf("\n");
 *     }
 * #endif
 *     MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, mybitstring, no_bs, MPI_INT,
 *             MPI_COMM_WORLD);
 *     setIntArrVal(col_update_order, no_global_cols, -1);
 *     int *tptr, *myptr;
 *     myptr = &mybitstring[no_bs*mypid];
 *     for (i = nprocs-1; i > 0 ; --i) {
 *         tptr = &mybitstring[no_bs*order[i]]; 
 *         for (j = 0; j < no_global_cols; ++j) {
 *             if(CHECK_BIT(tptr, j) && CHECK_BIT(myptr, j)){
 *                 col_update_order[j] = order[i];
 *             }
 *         }
 *     }
 * }
 */


void free_ucrows(ucrows *ucRows){
    if( ucRows->slcols != NULL)
        free(ucRows->slcols);
    if( ucRows->dslcols != NULL)
        free(ucRows->dslcols);
    if( ucRows-> col_update_order != NULL)
        free(ucRows-> col_update_order);
    if( ucRows->all_lcols != NULL)
        free(ucRows->all_lcols);
}

void init_ucrows(ucrows *ucRows, idx_t ngcols, idx_t nlcols, idx_t *lcols, int nprocs){
    idx_t lsum = 0, i;
    int *slcols, *dslcols;
    slcols = malloc(sizeof(*slcols) * nprocs);
    dslcols = calloc(nprocs, sizeof(*dslcols));
    int nllcols = nlcols;
    MPI_Allgather(&nllcols, 1, MPI_INT, slcols, 1, MPI_INT, MPI_COMM_WORLD);
    for (i = 1; i < nprocs; ++i) {
        dslcols[i] = dslcols[i-1] + slcols[i-1];
    }
    MPI_Allreduce(&nlcols, &lsum, 1, MPI_IDX_T, MPI_SUM, MPI_COMM_WORLD);
    ucRows->all_lcols = malloc(sizeof(*ucRows->all_lcols) * lsum);
    MPI_Allgatherv(lcols, nllcols, MPI_IDX_T, ucRows->all_lcols, slcols, dslcols, MPI_IDX_T, MPI_COMM_WORLD);
    ucRows->dslcols = dslcols;
    ucRows->slcols = slcols;
    ucRows->col_update_order = malloc(sizeof(*ucRows->col_update_order) * ngcols);
    setIntArrVal(ucRows->col_update_order, ngcols, -1);
    ucRows->no_global_cols = ngcols;
}

void get_shared_cols_efficient(ucrows *ucRows, const ldata *lData, const int *order) {
    node *head = NULL;
    node *current = NULL;
    node *prev;
    int sz = sizeof(int) * 8, nprocs, mypid ;
    idx_t no_bs = (ucRows->no_global_cols / sz) + 1, i, j, k;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
    idx_t  *all_lcols = ucRows->all_lcols;
#ifdef UGLY_DBG
    if (mypid == 0) {
        printf("Before Comm:\n");
        for (j = 0; j < no_bs; ++j) {
            printf("" PRINTF_BINARY_PATTERN_INT32 " ",
                    PRINTF_BYTE_TO_BINARY_INT32(mybitstring[mypid * no_bs + j]));
        }
        printf("\n");
    }
#endif


#ifdef NA_DBG_L2
    for (i = 0; i < nprocs; ++i) {
        na_log(dbgfp, "order[%d] = %d\n", i, order[i]);
    } 
#endif

    int *tmpbitstring = calloc(no_bs, sizeof(*tmpbitstring));
    int *tmpbitstring2 = calloc(no_bs, sizeof(*tmpbitstring2));
    int *cumulative_or_mask = calloc(no_bs, sizeof(*cumulative_or_mask));
    int *mbs = calloc(no_bs, sizeof(*mbs));

    for (i = 0; i < lData->nlcols; ++i) {
        setBit(mbs, lData->lcols[i]);
        node *tmp = malloc(sizeof(node));
        tmp->gidx = lData->lcols[i]; 
        tmp->next = head;
        head = tmp;
    }

    idx_t *tp = all_lcols + ucRows->dslcols[order[1]]; 
    for (i = 0; i < ucRows->slcols[order[1]]; ++i) {
        setBit(cumulative_or_mask, *(tp++));
    }
#ifdef UGLY_DBG
    if (mypid == 0) {
        printf("Before:\n");
        for (i = 0; i < nprocs; ++i) {
            for (j = 0; j < no_bs; ++j) {
                printf("" PRINTF_BINARY_PATTERN_INT32 " ",
                        PRINTF_BYTE_TO_BINARY_INT32(mybitstring[order[i] * no_bs + j]));
            }
            printf("\n");
        }
    }
#endif

    setIntArrVal(ucRows->col_update_order, ucRows->no_global_cols, -1);
    /* now compute for +1 */
    memcpy(tmpbitstring, cumulative_or_mask, sizeof(int) * no_bs);
    for (j = 0; j < no_bs; ++j) {
        tmpbitstring[j] &= mbs[j];
    }
    current = head;
    prev = NULL; 
    while(current != NULL){
        k = current->gidx;
        if(ucRows->col_update_order[k] == -1 && CHECK_BIT(tmpbitstring, k)){
            ucRows->col_update_order[k] = order[1];
            if(current == head)
                head = head->next;
            else
                prev->next = current->next;
            node *tmp = current;
            current = current->next;
            free(tmp);
        }
        else{
            prev = current;
            current = current->next;
        }
    }
    /* assign each column to the processor that first use it */
    for (i = 2; i < nprocs; ++i) {
        tp = all_lcols + ucRows->dslcols[order[i]]; 
        for (j = 0; j < ucRows->slcols[order[i]]; ++j) {
            setBit(tmpbitstring, *(tp));
            setBit(tmpbitstring2, *(tp++));
        }
        int * ptr = tmpbitstring2;
        for (j = 0; j < no_bs; ++j) {
            /* ucomm(px,py) = px & (py ^ (px&py & (px+1 | px+2 | ..... | py-1)) */
            ptr[j] = mbs[j] & (ptr[j] ^ ((mbs[j] & ptr[j]) & cumulative_or_mask[j]));
            // ptr++;
        }
        current = head;
        prev = NULL; 
        while(current != NULL){
            k = current->gidx;
            if(ucRows->col_update_order[k] == -1 && CHECK_BIT(ptr, k)){
                ucRows->col_update_order[k] = order[i];
                if(current == head)
                    head = head->next;
                else
                    prev->next = current->next;
                node *tmp = current;
                current = current->next;
                free(tmp);
            }
            else{
                prev = current;
                current = current->next;
            }
        }
        for (j = 0; j < no_bs; ++j) {
            cumulative_or_mask[j] |= tmpbitstring[j];
        }
    }
    /* cleanup */
    free(cumulative_or_mask);
    free(mbs);
    free(tmpbitstring);
    free(tmpbitstring2);
}

void get_shared_cols(triplet *M, idx_t nnz, idx_t no_global_cols, int *col_update_order, int *order) {

    int sz = sizeof(int) * 8, nprocs, mypid ;
    idx_t no_bs = (no_global_cols / sz) + 1, i, j;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
    int *mybitstring = calloc(no_bs * nprocs, sizeof(*mybitstring));
    for (i = 0; i < nnz; ++i) {
        assert(M[i].col < no_global_cols);
        setBit(mybitstring + (no_bs * mypid), M[i].col);
    }

#ifdef UGLY_DBG
    if (mypid == 0) {
        printf("Before Comm:\n");
        for (j = 0; j < no_bs; ++j) {
            printf("" PRINTF_BINARY_PATTERN_INT32 " ",
                    PRINTF_BYTE_TO_BINARY_INT32(mybitstring[mypid * no_bs + j]));
        }
        printf("\n");
    }
#endif
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, mybitstring, no_bs, MPI_INT,
            MPI_COMM_WORLD);


    int *tmpbitstring = malloc(sizeof(*tmpbitstring) * no_bs);
    int *cumulative_or_mask = calloc(no_bs, sizeof(*cumulative_or_mask));
    int *mbs = malloc(sizeof(*mbs) * no_bs);
#ifdef UGLY_DBG
    if (mypid == 0) {
        printf("Before:\n");
        for (i = 0; i < nprocs; ++i) {
            for (j = 0; j < no_bs; ++j) {
                printf("" PRINTF_BINARY_PATTERN_INT32 " ",
                        PRINTF_BYTE_TO_BINARY_INT32(mybitstring[order[i] * no_bs + j]));
            }
            printf("\n");
        }
    }
#endif
    memcpy(cumulative_or_mask, mybitstring + (no_bs * order[1]),
            sizeof(int) * no_bs);
    memcpy(mbs, mybitstring + (no_bs * mypid), sizeof(int) * no_bs);
    for (i = 2; i < nprocs; ++i) {
        memcpy(tmpbitstring, mybitstring + (no_bs * order[i]), sizeof(int) * no_bs);
        int *ptr = &mybitstring[order[i] * no_bs];
        for (j = 0; j < no_bs; ++j) {
            /* ucomm(px,py) = px & (py ^ (px&py & (px+1 | px+2 | ..... | py-1)) */
            ptr[j] = mbs[j] & (ptr[j] ^ ((mbs[j] & ptr[j]) & cumulative_or_mask[j]));
            // ptr++;
        }
        for (j = 0; j < no_bs; ++j) {
            cumulative_or_mask[j] |= tmpbitstring[j];
        }
    }

    /* now compute for +1 */
    for (j = 0; j < no_bs; ++j) {
        mybitstring[no_bs * order[1] + j] &= mybitstring[no_bs * mypid + j];
    }
#ifdef UGLY_DBG

    if (mypid == 0) {
        printf("\nafter\n");
        for (i = 0; i < nprocs; ++i) {
            for (j = 0; j < no_bs; ++j) {
                printf("" PRINTF_BINARY_PATTERN_INT32 " ",
                        PRINTF_BYTE_TO_BINARY_INT32(mybitstring[order[i] * no_bs + j]));
            }
            printf("\n");
        }
    }
#endif

    /* assign each column to the processor that first use it */
    setIntArrVal(col_update_order, no_global_cols, -1);
    int *tptr;
    for (i = 1; i < nprocs; ++i) {
        tptr = &mybitstring[no_bs*order[i]]; 
        for (j = 0; j < no_global_cols; ++j) {
            if(col_update_order[j] == -1 && CHECK_BIT(tptr, j)){
                col_update_order[j] = order[i];
            }
        }
    }
    /* cleanup */
    free(cumulative_or_mask);
    free(mbs);
    free(tmpbitstring);
    free(mybitstring);
}
// void get_shared_cols(triplet *M, idx_t nnz, idx_t no_global_cols)
//{
//
//    int sz = sizeof(int)*8, nprocs, mypid, *order;
//    idx_t no_bs = (no_global_cols / sz) + 1, i, j;
//    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
//    int *mybitstring = calloc(no_bs*nprocs, sizeof(*mybitstring));
//    for (i = 0; i < nnz; ++i) {
//        setBit(mybitstring+(no_bs*mypid), M[i].col);
//    }
//
//    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, mybitstring, no_bs,
//    MPI_INT, MPI_COMM_WORLD);
//
//    order = malloc(sizeof(*order) * nprocs);
//    get_ring_order(order, nprocs, mypid);
//
//    int *tmpbitstring = malloc(sizeof(*tmpbitstring) * no_bs);
//    int *cumulative_or_mask = calloc(no_bs, sizeof(*cumulative_or_mask));
//
//    memcpy(cumulative_or_mask, mybitstring+(no_bs * order[2]), no_bs);
//    for (i = 3; i < nprocs; ++i) {
//
//       memcpy(tmpbitstring, mybitstring+(no_bs * order[i]), no_bs);
//       idx_t *ptr = &mybitstring[order[i] *no_bs];
//       for (j = 0; j < no_bs; ++j) {
//          *ptr ^= mybitstring[no_bs*mypid+j] & (*ptr) & cumulative_or_mask[j];
//          ptr++;
//       }
//
//       for (j = 0; j < no_bs; ++j) {
//          cumulative_or_mask[j] |= tmpbitstring[j];
//       }
//    }
//}
