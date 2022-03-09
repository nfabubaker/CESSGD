/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : lData
 * @created     : Friday Nov 19, 2021 17:48:07 +03
 */
#include "basic.h"
#include "def.h"
#ifndef LDATA_H

#define LDATA_H

typedef struct _local_data{
    int nStms;
    int f;
    int nprocs;
    idx_t gnnz;
    idx_t nnz;
    idx_t *nnz_per_stratum;
    idx_t *nnz_per_row;
    idx_t *nnz_per_col;
    idx_t *nnz_per_row_l;
    idx_t *nnz_per_col_l;
    idx_t ngrows;
    idx_t nlrows;
    idx_t ngcols; /* # global cols */
    idx_t nlcols; /* # of local cols */
    idx_t maxColStrip;
    idx_t *xgcols;
    idx_t *xlcols; /* local cols per stratum */
    idx_t *lcols; /* stores global col id per local col */
    idx_t *gtlcolmap; /* map of global to local cols */
    idx_t *gtlrowmap;
    idx_t *gtgcolmap; /* map global to global in ordered fashion for naive method*/
    triplet **mtx; /* local matrix entries per stratum */
    real_t *qmat;
    real_t *rmat;

} ldata;

void init_lData(ldata *lData);
void free_lData(ldata *lData);
void setup_ldata(const triplet *M, ldata *lData, const int *partvec, const int comm_type, const int f);


#endif /* end of include guard LDATA_H */
