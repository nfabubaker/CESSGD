/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : sgd_seq
 * @created     : Tuesday Aug 04, 2020 17:42:48 +03
 */

#include <stdlib.h>
#include "basic.h"
#include "util.h"
#include "sgd.h"
#include <math.h>
#include <time.h>
#include <stdarg.h>

void na_log(FILE *fp, const char* format, ... ){
    va_list args;
    va_start(args, format);
    vfprintf(fp, format, args);
    fflush(fp);
    va_end(args);

}


void Read_Matrix(char* filename, triplet **M, idx_t *nrows, idx_t *ncols, idx_t *nnz, idx_t **nnz_per_row, idx_t **nnz_per_col){

    FILE *fp = fopen(filename, "r");
    char line[1024];
    int cond =1;
    while(cond){
        fgets(line, 1024, fp);
        if(line[0] != '%'){
            sscanf(line,"%u %u %u\n", nrows, ncols, nnz);
            cond = 0;
            break;
        }
    }
    *M = malloc(sizeof(triplet) * (*nnz));
    (*nnz_per_row) = malloc((*nrows) * sizeof(idx_t));
    (*nnz_per_col) = malloc((*ncols) * sizeof(idx_t));
    setIDXTArrZero(*nnz_per_row, *nrows);
    setIDXTArrZero(*nnz_per_col, *ncols);
    rewind(fp);
    do {
        fgets(line, 1024, fp);

    } while (line[0]=='%');
    int pid;
    triplet tt;
    idx_t i;
    for (i = 0; i < *nnz; ++i) {
        fgets(line, 128, fp);
        sscanf(line, "%u %u %lf\n", &tt.row, &tt.col, &tt.val );
#ifdef NA_DBG
        if(!i)
            na_log(dbgfp, "\t 2x read first line %zu %zu %lf\n", tt.row, tt.col, tt.val);
        assert(trow <= gs->ngrows);
#endif
        tt.row--; tt.col--; //make col and row inds start from 0
        ((*nnz_per_row)[tt.row])++;
        ((*nnz_per_col)[tt.col])++;
        (*M)[i]  = tt;
    }
    fclose(fp);
}

void Read_info(char *filename, idx_t *nnz, idx_t *nrows, idx_t *ncols){
    FILE *fp;

    fp = fopen(filename, "rb");
    fread(nrows, sizeof(int), 1, fp);
    fread(ncols, sizeof(int), 1, fp);
    fseek(fp, sizeof(int) * 6, SEEK_SET);
    fread(nnz, sizeof(int), 1, fp);
    fclose(fp);
}

int main(int argc, char *argv[])
{

#ifdef NA_DBG
    fprintf(stderr, "Hello from main1\n");
#endif

    char  mFN[1024];
    idx_t i, nnz, nrows, ncols ;
    real_t *qmat, *rmat; 
    sgd_params params;
    double stepSize, initLoss, finalLoss;
    triplet *M;
    idx_t niter = 10, nwarmup = 2, f = 16;

#ifdef NA_DBG
    fprintf(stderr, "Hello from main2\n");
#endif
    if(argc < 2){
        printf("usage: %s matrixFile factorizationRank\n", argv[0]);
        exit(0);
    }
    sprintf(mFN, "%s", argv[1]);
    f = atoi(argv[2]);
    /* Timing purposes */
#ifdef TAKE_TIMES
    tmr_t setuptime, sgdtime;
    setuptime.elapsed = 0;
    sgdtime.elapsed = 0;
    start_timer(&setuptime);
#endif

#ifdef NA_DBG
    fprintf(stderr, "Hello from main\n");
    sprintf(dbg_fn, "../res/seq-outfile");
    dbgfp = fopen(dbg_fn, "w");
    na_log(dbgfp, "> Inside main\nJust before Initialize\n");
#endif 

    idx_t *nnz_per_row, *nnz_per_col; 
    Read_Matrix(mFN, &M, &nrows, &ncols, &nnz, &nnz_per_row, &nnz_per_col);
#ifdef NA_DBG
    fprintf(stderr, "after read matrix\n");
#endif
    qmat = malloc(ncols * f * sizeof(*qmat));
    rmat = malloc(sizeof(*rmat) * nrows * f);
    mat_init(qmat, ncols*f, f);
    mat_init(rmat, nrows*f, f);

#ifdef NA_DBG
    fprintf(stderr, "after init qmat and rmat\n");
    fprintf(stderr, "M[0].row=%d, M[0].col=%d, M[0].val=%f\n",M[0].row, M[0].col, M[0].val);
#endif

#ifdef TAKE_TIMES
    stop_timer(&setuptime);
    start_timer(&sgdtime);
#endif
/*     for (i = 0; i < nwarmup; ++i) {
 *         _sgd_l(rmat, qmat, f, nnz, M, stepSize, lambda); 
 *     }
 * #ifdef NA_DBG
 *     fprintf(stderr, "after warmup, error =%f \n", err);
 * #endif
 * #ifdef TAKE_TIMES
 *     begin = clock();
 *     wutime = (double) (begin - end) / CLOCKS_PER_SEC;
 * #endif
 */
    params.init_eps = 0.0075; params.lambda = 0.05; params.eps_inc = 1.05; params.eps_dec = 0.5;
    double prevLoss, currLoss;
    initLoss = compute_loss(rmat, qmat, f, nnz, M, nrows, ncols , nnz_per_row, nnz_per_col, params.lambda);
    currLoss = initLoss;
    for (i = 0; i < niter; ++i) {
        if(i == 0)
            stepSize = params.init_eps;
        else
            stepSize = update_stepSize(&params, stepSize, prevLoss, currLoss);
        _sgd_l(rmat, qmat, f, nnz, M, stepSize, params.lambda); 
        prevLoss = currLoss;
        currLoss = compute_loss(rmat, qmat, f, nnz, M, nrows, ncols, nnz_per_row, nnz_per_col, params.lambda);
        printf("[i%u] eps=%f prevLoss=%f, currLoss=%f\n",i, stepSize, prevLoss, currLoss);
    }
    finalLoss = currLoss;

#ifdef TAKE_TIMES
    stop_timer(&sgdtime);
    char fname[1024], name[1024];
    substring(fname, mFN);
    substring_b(name, fname);
    double mult = 1e-6;
    printf("%s %d %d %.5f %.5f %.5f %.5f %.5f\n", name, 1 , f, initLoss, finalLoss, initLoss-finalLoss , setuptime.elapsed*mult, (mult*sgdtime.elapsed)/(real_t)niter);
#endif
    free(qmat); free(rmat); free(nnz_per_row); free(nnz_per_col); 
    return 0;
}
