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


#ifndef _IO_H
#define _IO_H

#include "def.h"
#include "lData.h"

void read_metadata(const char* filename, idx_t *ngrows, idx_t *ngcols, idx_t *gnnz);
void read_matrix_bc(const char *mFN, triplet **mtx, const int * const rpvec, ldata *lData);
void read_partvec_bc(const char *pvecFN, int * const rpvec, int * const colpvec, const int ngrows, const int ngcols, const int use_randColDist);
//void read_matrix(const char* filename, triplet **M, genst *gs);
//void read_offsets(const char* filename, genst *gs);
//void read_row_col_idxs(const char* filename, genst *gs);
//void read_partvec(const char* filename, genst *gs);
#endif /* end of include guard: _IO_H */
