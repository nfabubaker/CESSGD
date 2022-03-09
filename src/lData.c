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
#include "lData.h"
#include "sgd.h"
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>


void init_lData(ldata *lData){
    
    lData->nnz_per_stratum = NULL;
    lData->nnz_per_row = NULL;
    lData->nnz_per_col = NULL;
    lData->nnz_per_row_l = NULL;
    lData->nnz_per_col_l = NULL;
    lData->xgcols = NULL;
    lData->xlcols = NULL; /* local cols per stratum */
    lData->lcols = NULL; /* stores global col id per local col */
    lData->gtlcolmap = NULL; /* map of global to local cols */
    lData->gtlrowmap = NULL;
    lData->gtgcolmap = NULL; /* map global to global in ordered fashion for naive method*/
    lData->mtx = NULL; /* local matrix entries per stratum */
    lData->qmat = NULL;
    lData->rmat = NULL;
}

void free_lData(ldata *lData){
    if( lData->nnz_per_stratum != NULL ){
        free( lData->nnz_per_stratum );
        lData->nnz_per_stratum = NULL;
    }
    if( lData->nnz_per_row_l != NULL ){
        free( lData->nnz_per_row_l );
        lData->nnz_per_row_l = NULL;
    }
    if( lData->nnz_per_col_l != NULL ){
        free( lData->nnz_per_col_l );
        lData->nnz_per_col_l = NULL;
    }
    if( lData->xgcols != NULL ){
        free( lData->xgcols );
        lData->xgcols = NULL;
    }
    if( lData->xlcols != NULL ){ /* local cols per stratum */
        free( lData->xlcols );
        lData->xlcols = NULL;
    }
    if( lData->lcols != NULL ){ /* stores global col id per local col */
        free( lData->lcols );
        lData->lcols = NULL;
    }
    if( lData->gtlcolmap != NULL ){ /* map of global to local cols */
        free( lData->gtlcolmap );
        lData->gtlcolmap = NULL;
    }
    if( lData->gtlrowmap != NULL ){
        free( lData->gtlrowmap );
        lData->gtlrowmap = NULL;
    }
    if( lData->gtgcolmap != NULL ){ /* map global to global in ordered fashion for naive method*/
        free( lData->gtgcolmap );
        lData->gtgcolmap = NULL;
    }
    if( lData->mtx != NULL ){ /* local matrix entries per stratum */
        int i;
        for(i = 0; i < lData->nprocs; i++)
            if(lData->mtx[i] != NULL){
                free(lData->mtx[i]);
                lData->mtx[i] = NULL;
            }
        free( lData->mtx );
        lData->mtx = NULL;
    }
    if( lData->qmat != NULL ){
        free( lData->qmat );
        lData->qmat = NULL;
    }
    if( lData->rmat != NULL ){
        free( lData->rmat );
        lData->rmat = NULL;
    }

}

void init_local_inds(triplet **mtx, const int* partvec, ldata *gs)
{
    idx_t *xlcols, *lcols, *gcols, *grows, i, j; 
    gs->xlcols = calloc((gs->nprocs+2), sizeof(*gs->xlcols));
    xlcols = gs->xlcols;
    gs->gtlcolmap = malloc(gs->ngcols * sizeof(*gcols));
    gs->gtlrowmap = malloc(gs->ngrows * sizeof(*grows));
    gcols = gs->gtlcolmap;
    grows = gs->gtlrowmap;
    setIDXTArrVal(gcols, gs->ngcols, -1);
    setIDXTArrVal(grows, gs->ngrows, -1);
    for (i = 0; i < gs->nprocs; ++i) {
        for (j = 0; j < gs->nnz_per_stratum[i]; ++j) {
            gcols[mtx[i][j].col] = 1;
            grows[mtx[i][j].row] = 1;
        }
    }
    /* count local cols and rows */
    gs->nlcols= gs->nlrows = 0;
    /* FIXME: remove this?, gs->nlcols is last idx of xlcols */
    for (i = 0; i < gs->ngcols; ++i){
        if (gcols[i] != -1) {
            gs->nlcols++;
        }
    }
    for (i = 0; i < gs->ngrows; ++i) 
        if(grows[i] != -1)
            grows[i] = gs->nlrows++;

    for (i = 0; i < gs->ngcols; ++i) {
        if(gcols[i] != -1)
            xlcols[partvec[i]+2]++; 
    }
    for (i = 2; i < gs->nprocs+2; ++i) {
        xlcols[i] += xlcols[i-1]; 
    }
    gs->nlcols = xlcols[gs->nprocs+1];
    gs->lcols = malloc(sizeof(*gs->lcols) * gs->nlcols);
    lcols = gs->lcols;
    for (i = 0; i < gs->ngcols; ++i) {
        if(gcols[i] != -1){
            lcols[xlcols[partvec[i]+1]] = i;
            gcols[i] = xlcols[partvec[i]+1]++;
        }
    }
    assert(xlcols[gs->nprocs] == xlcols[gs->nprocs+1]);
    assert(xlcols[gs->nprocs] == gs->nlcols);
    for (i = 0; i < gs->nprocs; ++i) {
        for (j = 0; j < gs->nnz_per_stratum[i]; ++j) {
            mtx[i][j].col = gcols[mtx[i][j].col];
            mtx[i][j].row = grows[mtx[i][j].row];
        }
    }
}

void matrix_to_perStratum(const triplet *M, const int *partvec, ldata *gs){
    idx_t i, *cnts;
    /* count how many nnz per stratum */
    gs->nnz_per_stratum = calloc(gs->nprocs, sizeof(*gs->nnz_per_stratum));
    cnts = gs->nnz_per_stratum;
    for (i = 0; i < gs->nnz; ++i) {
        cnts[partvec[M[i].col]]++;
    }
    /* allocate */
    for (i = 0; i < gs->nprocs; ++i) {
        gs->mtx[i] = malloc(sizeof(**gs->mtx) * cnts[i]);
        cnts[i] = 0;
    }
    /* copy triplets to corresponding SE */
    int part;
    for (i = 0; i < gs->nnz; ++i) {
        part = partvec[M[i].col];
        gs->mtx[part][cnts[part]++] = M[i];
    }
}

void init_naive(const triplet *M, const int *partvec, ldata *gs){
    idx_t i, j;
    gs->maxColStrip = 0;
    /* convert input matrix to per stratum */
    gs->mtx = malloc(sizeof(*gs->mtx) * gs->nprocs);
    matrix_to_perStratum(M, partvec, gs);
    /* count local cols and rows */
    idx_t *grows;
    gs->gtlrowmap = malloc(gs->ngrows * sizeof(*grows));
    grows = gs->gtlrowmap;
    /* convert input matrix to per stratum */
    setIDXTArrVal(grows, gs->ngrows, -1);
    for (i = 0; i < gs->nprocs; ++i) {
        for (j = 0; j < gs->nnz_per_stratum[i]; ++j) {
            grows[gs->mtx[i][j].row] = 1;
        }
    }
    gs->nlrows = 0;
    for (i = 0; i < gs->ngrows; ++i) 
        if(grows[i] != -1)
            grows[i] = gs->nlrows++;
    for (i = 0; i < gs->nprocs; ++i) {
        for (j = 0; j < gs->nnz_per_stratum[i]; ++j) {
            gs->mtx[i][j].row = grows[gs->mtx[i][j].row];
            //gs->mtx[i][j].col %= 512; /* TODO FIXME remove this */
        }
    }
    gs->xlcols = calloc((gs->nprocs+2), sizeof(*gs->xlcols));
    for (i = 0; i < gs->ngcols; ++i) {
        gs->xlcols[partvec[i]+2]++;
    }
    for (i = 2; i < gs->nprocs+2; ++i) {
        if(gs->xlcols[i] > gs->maxColStrip)
            gs->maxColStrip = gs->xlcols[i];
        gs->xlcols[i]+= gs->xlcols[i-1];
    }
    assert(gs->xlcols[gs->nprocs+1] == gs->ngcols);
    gs->lcols = calloc(gs->ngcols, sizeof(*gs->lcols));
    for (i = 0; i < gs->ngcols; ++i) {
        gs->lcols[gs->xlcols[partvec[i]+1]++] = i;
    }
    assert(gs->xlcols[gs->nprocs] == gs->xlcols[gs->nprocs+1]);
    gs->qmat = malloc(sizeof(*gs->qmat) * gs->ngcols * gs->f);
    gs->rmat = malloc(sizeof(*gs->rmat) * gs->nlrows * gs->f);
    mat_init(gs->qmat, gs->ngcols*gs->f, gs->f);
    mat_init(gs->rmat, gs->nlrows*gs->f, gs->f);
}

void init_naive_new(const triplet *M, const int *partvec, ldata *gs){
    idx_t i,j, maxColStrip = 0;
    idx_t *grows, *gcols;
    gs->gtlrowmap = malloc(gs->ngrows * sizeof(*grows));
    gs->gtlcolmap = malloc(sizeof(*gs->gtlcolmap) * gs->ngcols);
    gs->gtgcolmap = malloc(sizeof(*gs->gtgcolmap) * gs->ngcols);
    grows = gs->gtlrowmap;
    gcols = gs->gtlcolmap;
    /* convert input matrix to per stratum */
    gs->mtx = malloc(sizeof(*gs->mtx) * gs->nprocs);
    matrix_to_perStratum(M, partvec, gs);
    setIDXTArrVal(grows, gs->ngrows, -1);
    setIDXTArrVal(gcols, gs->ngcols, -1);
    for (i = 0; i < gs->nprocs; ++i) {
        for (j = 0; j < gs->nnz_per_stratum[i]; ++j) {
            grows[gs->mtx[i][j].row] = 1;
            gcols[gs->mtx[i][j].col] = 1;
        }
    }
    /* count local cols and rows */
    gs->nlrows = 0;
    gs->nlcols = 0;
    for (i = 0; i < gs->ngrows; ++i) 
        if(grows[i] != -1)
            grows[i] = gs->nlrows++;
    for (i = 0; i < gs->ngcols; ++i) 
        if(gcols[i] != -1)
            gcols[i] = gs->nlcols++;

    for (i = 0; i < gs->nprocs; ++i) {
        for (j = 0; j < gs->nnz_per_stratum[i]; ++j) {
            gs->mtx[i][j].row = grows[gs->mtx[i][j].row];
            gs->mtx[i][j].col = gcols[gs->mtx[i][j].col];
        }
    }
    gs->xlcols = calloc((gs->nprocs+2), sizeof(*gs->xlcols));
    gs->xgcols = calloc((gs->nprocs+2), sizeof(*gs->xgcols));

    for (i = 0; i < gs->ngcols; ++i) {
        gs->xgcols[partvec[i]+2]++;
        if(gcols[i] != -1)
            gs->xlcols[partvec[i]+2]++;
    }
    for (i = 2; i < gs->nprocs+2; ++i) {
        if(gs->xgcols[i] > maxColStrip)
            maxColStrip = gs->xgcols[i];
        gs->xlcols[i]+= gs->xlcols[i-1];
        gs->xgcols[i]+= gs->xgcols[i-1];
    }
    assert(gs->xlcols[gs->nprocs+1] == gs->nlcols);
    assert(gs->xgcols[gs->nprocs+1] == gs->ngcols);
    gs->lcols = calloc(gs->nlcols, sizeof(*gs->lcols));
    for (i = 0; i < gs->ngcols; ++i) {
        gs->gtgcolmap[i] = gs->xgcols[partvec[i]+1]++;
        if(gcols[i] != -1)
            gs->lcols[gs->xlcols[partvec[i]+1]++] = i;
    }
    assert(gs->xlcols[gs->nprocs] == gs->xlcols[gs->nprocs+1]);
    gs->qmat = malloc(sizeof(*gs->qmat) * gs->nlcols * gs->f);
    gs->rmat = malloc(sizeof(*gs->rmat) * gs->nlrows * gs->f);
    mat_init(gs->qmat, gs->nlcols*gs->f, gs->f);
    mat_init(gs->rmat, gs->nlrows*gs->f, gs->f);
}

void setup_ldata(const triplet *M, ldata *lData, const int *partvec, const int comm_type, const int f) {
    int nprocs, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    lData->nprocs = nprocs;
    lData->nStms = nprocs;
    lData->f = f;
    if(comm_type == NAIIVE){
        init_naive(M, partvec, lData);
        return;
    }
    idx_t i;

    /* convert input matrix to per stratum */
    lData->mtx = malloc(sizeof(*lData->mtx) * nprocs);
    matrix_to_perStratum(M, partvec, lData);
    init_local_inds(lData->mtx, partvec, lData);

    lData->nnz_per_row_l = malloc(lData->nlrows * sizeof(*lData->nnz_per_row_l));
    lData->nnz_per_col_l = malloc(lData->nlcols * sizeof(*lData->nnz_per_col_l));
    
    for (i = 0; i < lData->ngrows; ++i) {
       if(lData->gtlrowmap[i] != -1){
           lData->nnz_per_row_l[lData->gtlrowmap[i]] = lData->nnz_per_row[i];  
       }
    } 

    for (i = 0; i < lData->ngcols; ++i) {
       if(lData->gtlcolmap[i] != -1){
           lData->nnz_per_col_l[lData->gtlcolmap[i]] = lData->nnz_per_col[i];  
       }
    } 

    /* init factor matrices */

    lData->qmat = malloc(sizeof(*lData->qmat) * lData->nlcols * lData->f);
    lData->rmat = malloc(sizeof(*lData->rmat) * lData->nlrows * lData->f);
    mat_init(lData->qmat, lData->nlcols*lData->f, lData->f);
    mat_init(lData->rmat, lData->nlrows*lData->f, lData->f);
}
