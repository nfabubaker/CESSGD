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

#include "io.h"
#include "basic.h"
#include "def.h"
#include <mpi.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
/*! \brief reads 
 *
 *  Detailed description of the function
 *
 * \param filename Parameter description
 * \param gs Parameter description
 * \return Return parameter description
 */
void read_metadata(const char* filename, idx_t *ngrows, idx_t *ngcols, idx_t *gnnz)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    idx_t tarr[3];
    if(rank == 0){
        FILE *fp = fopen(filename, "r");
        char line[1024];
        int cond =1;
        while(cond){
            fgets(line, 1024, fp);
            if(line[0] != '%'){
#if idxsize == 64
                sscanf(line,"%lu %lu %lu\n", &tarr[0], &tarr[1], &tarr[2]);
#elif idxsize == 32
                sscanf(line,"%u %u %u\n", &tarr[0], &tarr[1], &tarr[2]);
#endif

                cond = 0;
                break;
            }
        }
        fclose(fp);
    }

    MPI_Bcast(tarr, 3, MPI_IDX_T, 0, MPI_COMM_WORLD);
    *ngrows = tarr[0];
    *ngcols = tarr[1];
    *gnnz = tarr[2];
}

void read_partvec_bc(const char *pvecFN, int *const rpvec, int *const colpvec, const int ngrows, const int ngcols,const  int use_randColDist)
{
    int rank, i;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank == 0){
        FILE *fp = fopen(pvecFN, "r");
        char line[128];
        for (i = 0; i < ngrows; ++i) {
            fgets(line, 128, fp);
            sscanf(line, "%d\n", &rpvec[i]);
        }
        if(use_randColDist == 0){
            fgets(line, 128, fp);
            assert(line[0]=='-');
            for (i = 0; i < ngcols; ++i) {
                fgets(line, 128, fp);
                sscanf(line, "%d\n", &colpvec[i]);
            }
        }
        fclose(fp);
    }   
    MPI_Bcast(rpvec, ngrows, MPI_INT, 0, MPI_COMM_WORLD);
    if(use_randColDist == 0)
        MPI_Bcast(colpvec, ngcols, MPI_INT, 0, MPI_COMM_WORLD);
}

void read_matrix_bc(const char *mtxFN, triplet **mtx, const int * const rpvec, ldata *lData)
{

    size_t i;
    idx_t *nnzcnts = NULL;
    triplet **M; 
    /* create a type for struct triplet */
    const int nitems = 3;
    int myrank, nprocs ,blocklengths[3] = {1, 1, 1};
    MPI_Datatype types[3] = {MPI_IDX_T, MPI_IDX_T, MPI_REAL_T};
    MPI_Datatype mpi_s_type;
    MPI_Aint offsets[3];
    offsets[0] = offsetof(triplet, row);
    offsets[1] = offsetof(triplet, col);
    offsets[2] = offsetof(triplet, val);
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_s_type);
    MPI_Type_commit(&mpi_s_type);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    /* new datatype committed*/

    lData->nnz_per_row = malloc(lData->ngrows * sizeof(*lData->nnz_per_row));
    lData->nnz_per_col = malloc(lData->ngcols * sizeof(*lData->nnz_per_col));
    if(myrank == 0){
        FILE *fp = fopen(mtxFN, "r");
        idx_t trow, tcol;
        real_t tval;
        lData->nnz = 0;
        char line[1024];
        M = malloc(nprocs * sizeof(*M));
        nnzcnts = calloc(nprocs, sizeof(*nnzcnts));
        setIDXTArrZero(lData->nnz_per_row, lData->ngrows);
        setIDXTArrZero(lData->nnz_per_col, lData->ngcols);
        do {
            fgets(line, 1024, fp);

        } while (line[0]=='%');
        for (i = 0; i < lData->gnnz; ++i) {
            fgets(line, 128, fp);
#if idxsize == 64
            sscanf(line, "%lu %lu %lf\n", &trow, &tcol, &tval );
#elif idxsize == 32
            sscanf(line, "%u %u %lf\n", &trow, &tcol, &tval );
#endif
            trow--; tcol--;
            lData->nnz_per_row[trow]++;
            lData->nnz_per_col[tcol]++;

            if(rpvec[trow] == 0){
                lData->nnz++;
            }
            else{
                nnzcnts[rpvec[trow]]++;
            }
        }

        /* create my own mtx */
        *mtx = malloc(lData->nnz * sizeof(**mtx));
        for (i = 1; i < nprocs; ++i) {
            M[i] = malloc(nnzcnts[i]*sizeof(*M[i]));
            nnzcnts[i] = 0;
        }
        rewind(fp);
        do {
            fgets(line, 1024, fp);

        } while (line[0]=='%');
        int pid;
        triplet tt;
        lData->nnz = 0;
        for (i = 0; i < lData->gnnz; ++i) {
            fgets(line, 128, fp);
#if idxsize == 64
            sscanf(line, "%lu %lu %lf\n", &tt.row, &tt.col, &tt.val );
#elif idxsize == 32
            sscanf(line, "%u %u %lf\n", &tt.row, &tt.col, &tt.val );
#endif
            tt.row--; tt.col--; //make col and row inds start from 0
            pid = rpvec[tt.row];
            if(pid == 0)
                (*mtx)[lData->nnz++] = tt;
            else
                M[pid][nnzcnts[pid]++] = tt;
        }


        nnzcnts[0] = lData->nnz;
    }

    MPI_Bcast(lData->nnz_per_col, lData->ngcols, MPI_IDX_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(lData->nnz_per_row, lData->ngrows, MPI_IDX_T, 0, MPI_COMM_WORLD);
    /* scatter the counts */
    MPI_Scatter(nnzcnts, 1, MPI_IDX_T, &lData->nnz, 1, MPI_IDX_T, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        for (i = 1; i < nprocs; ++i) {
            MPI_Send(M[i], nnzcnts[i], mpi_s_type, i, 11, MPI_COMM_WORLD);
        }
        for (i = 1; i < nprocs; ++i) {
            free(M[i]);
        }
        free(M);
        free(nnzcnts);
    }
    else{
        /* allocate my own matrix */
        *mtx = malloc(lData->nnz * sizeof(**mtx));
        MPI_Recv((*mtx), lData->nnz, mpi_s_type, 0, 11, MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
    }

}

