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
