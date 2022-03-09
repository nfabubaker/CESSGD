
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
