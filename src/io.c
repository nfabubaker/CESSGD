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

#ifdef NA_DBG
    na_log(dbgfp, "> In read matrix with bcast\n");
#endif
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
#ifdef NA_DBG
            if(!i)
                na_log(dbgfp, "\tread first line %u %u %lf\n", trow, tcol, tval);
            assert(trow < lData->ngrows);
#endif

            if(rpvec[trow] == 0){
                lData->nnz++;
            }
            else{
#ifdef NA_DBG
                assert(rpvec[trow] < nprocs);
#endif
                nnzcnts[rpvec[trow]]++;
            }
        }

#ifdef NA_DBG
        na_log(dbgfp, "\tdone counting nnzeros per processors\n");
#endif
        /* create my own mtx */
        *mtx = malloc(lData->nnz * sizeof(**mtx));
        for (i = 1; i < nprocs; ++i) {
            M[i] = malloc(nnzcnts[i]*sizeof(*M[i]));
            nnzcnts[i] = 0;
        }
#ifdef NA_DBG
        na_log(dbgfp, "\tdone create my own mtx\n");
#endif
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
#ifdef NA_DBG
            if(!i)
                na_log(dbgfp, "\t 2x read first line %u %u %lf\n", tt.row, tt.col, tt.val);
            assert(trow <= lData->ngrows);
#endif
            tt.row--; tt.col--; //make col and row inds start from 0
            pid = rpvec[tt.row];
            if(pid == 0)
                (*mtx)[lData->nnz++] = tt;
            else
                M[pid][nnzcnts[pid]++] = tt;
        }

#ifdef NA_DBG
        na_log(dbgfp, "\tdone filling other processors' nonzeros\n");
#endif

        nnzcnts[0] = lData->nnz;
    }

    MPI_Bcast(lData->nnz_per_col, lData->ngcols, MPI_IDX_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(lData->nnz_per_row, lData->ngrows, MPI_IDX_T, 0, MPI_COMM_WORLD);
    /* scatter the counts */
    MPI_Scatter(nnzcnts, 1, MPI_IDX_T, &lData->nnz, 1, MPI_IDX_T, 0, MPI_COMM_WORLD);

#ifdef NA_DBG

    na_log(dbgfp, "\tdone scatter\n");
#endif
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

#ifdef NA_DBG
    na_log(dbgfp, "\tread mtx done\n");
#endif
}

/******************************************************************************
 * Function:         void readMatrix
 * Description:
 * Where:
 * Return:
 * Error:
 *****************************************************************************/
//void read_matrix(const char *filename, triplet **mat, genst *gs) {
//  #ifdef NA_DBG
//      na_log(dbgfp, "> in read_matrix\n matrix fn: %s\n", filename);
//  #endif
//    
//  MPI_File fh;
//  MPI_Status s;
//  MPI_Comm comm = MPI_COMM_WORLD;
//
//  /* create a type for struct triplet */
//  const int nitems = 3;
//  int blocklengths[3] = {1, 1, 1};
//  MPI_Datatype types[3] = {MPI_INT, MPI_INT, MPI_DOUBLE};
//  MPI_Datatype mpi_s_type;
//  MPI_Aint offsets[3];
//  offsets[0] = offsetof(triplet, row);
//  offsets[1] = offsetof(triplet, col);
//  offsets[2] = offsetof(triplet, val);
//  MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_s_type);
//  MPI_Type_commit(&mpi_s_type);
//  /* new datatype committed*/
//
//  *mat = malloc(gs->nnz * sizeof(triplet));
//
//  /* NABIL TODO : do i need the following ? */
//  //    double * avg_r = calloc(gs->rows_owned, sizeof(double));
//  ////    double * avg_q_lcl = calloc(100, sizeof(double));
//  ////    double * avg_q_glb= calloc(100, sizeof(double));
//  //    int * avg_r_cnt = calloc(gs->rows_owned, sizeof(int));
//  ////    int * avg_q_cnt_lcl = calloc(100, sizeof(int));
//  ////    int * avg_q_cnt_glb= calloc(100, sizeof(int));
//  //    double avg_rtg = 0;
//
//  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//  MPI_File_seek(fh, sizeof(triplet) * (gs->start_offset), MPI_SEEK_SET);
//  MPI_File_read(fh, *mat, gs->nnz, mpi_s_type, &s);
//
///*   int i;
// *   for (i = 0; i < gs->nnz; i++) { // seems like a bad practice
// *     MPI_File_seek(fh, sizeof(triplet) *  gs->start_offset, MPI_SEEK_SET);
// *     MPI_File_read(fh, &mat[i], 1, mpi_s_type, &s);
// *     //        avg_rtg+= mat[i].val;
// *   }
// */
//
//
//  //avg_rtg /= nnz;
//  //
//  MPI_File_close(&fh);
//}

//void read_offsets(const char *fn, genst *gs) {
//    #ifdef NA_DBG
//        na_log(dbgfp, "> in read_offset\noffset fn = %s\n", fn);
//    #endif
//  MPI_Comm comm = MPI_COMM_WORLD;
//  MPI_Status s;
//
//  int rank = gs->myrank;
//
//  MPI_File fh;
//  // read files
//  MPI_File_open(comm, fn, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//  MPI_File_seek(fh, sizeof(int) * (11 * (rank)), MPI_SEEK_SET);
//  MPI_File_read(fh, &gs->glbRowCnt, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->glbColCnt, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->lclColCnt, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->ownerColCnt, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->nonownerColCnt, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->rowcolOffset, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->nnz, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->start_offset, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->rows_owned, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->ql_starting_index, 1, MPI_INT, &s);
//  MPI_File_read(fh, &gs->ql_ownerCnt, 1, MPI_INT, &s);
//  MPI_File_close(&fh);
//
//  gs->nonownerColCnt -= gs->ownerColCnt;
//  gs->ownerColCnt -= gs->lclColCnt;
//  gs->qlColCnt = gs->nonownerColCnt + gs->ownerColCnt;
//  gs->totalColCnt = gs->lclColCnt + gs->ownerColCnt + gs->nonownerColCnt;
//
//  #ifdef NA_DBG
//      na_log(dbgfp, "GS status:\ngs->nnz=%d, gs->glbRowCnt=%d, gs->glbColCnt=%d, gs->ownerColCnt=%d, gs->nonownerColCnt=%d\n", gs->nnz, gs->glbRowCnt, gs->glbColCnt, gs->ownerColCnt, gs->nonownerColCnt); 
//  #endif
//}
//
//void read_row_col_idxs(const char* filename, genst *gs) {
//    MPI_File fh;
//    MPI_Status s;
//    MPI_Comm comm = MPI_COMM_WORLD;
//  gs->q_idxs_full = malloc(gs->totalColCnt *sizeof(*gs->q_idxs_full)); // global indexes of all columns(no comm + comm ones)
//  //double *avg_q = calloc( totalColCnt, sizeof(double));   
// // gs->avg_q_cnt = calloc(totalColCnt * sizeof(int)); 
//
//  gs->row_mapping = malloc(gs->rows_owned * sizeof(*gs->row_mapping));
//
//  MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//  MPI_File_seek(fh, sizeof(int) * (gs->rowcolOffset), MPI_SEEK_SET);
//  MPI_File_read(fh, gs->row_mapping, gs->rows_owned, MPI_INT, &s);
//  MPI_File_seek(fh, sizeof(int) * (gs->rowcolOffset + gs->rows_owned), MPI_SEEK_SET);
//  MPI_File_read(fh, gs->q_idxs_full, gs->totalColCnt, MPI_INT, &s);
//
//  MPI_File_close(&fh);
//}
//
//void read_partvec(const char* filename, genst *gs) {
//    #ifdef NA_DBG
//        na_log(dbgfp, "> in read_partvec\n partvec fn: %s\n", filename);
//    #endif
//    MPI_File fh;
//    MPI_Status s;
////    MPI_Comm comm = MPI_COMM_WORLD;
//    MPI_Comm comm = MPI_COMM_SELF;
//    gs->partvec = malloc(gs->glbColCnt *sizeof(*gs->partvec)); // global indexes of all columns(no comm + comm ones)
//    //double *avg_q = calloc( totalColCnt, sizeof(double));
//    // gs->avg_q_cnt = calloc(totalColCnt * sizeof(int));
//    MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
//    MPI_File_seek(fh, 0, MPI_SEEK_SET);
//    MPI_File_read(fh, gs->partvec, gs->glbColCnt, MPI_INT, &s);
//    MPI_File_close(&fh);
//    #ifdef NA_DBG
//        na_log(dbgfp, "> in read_partvec\n finished read...\n");
//    #endif
//
//}

