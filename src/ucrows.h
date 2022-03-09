/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : ucrows
 * @created     : Friday Nov 19, 2021 18:09:49 +03
 */

#ifndef UCROWS_H

#define UCROWS_H
#include "basic.h"
#include "def.h"
#include "lData.h"

typedef struct _ucrows{
    int *col_update_order;
    idx_t *all_lcols;
    int *dslcols, *slcols;
    idx_t nnz, no_global_cols;

    
} ucrows;

void get_shared_cols_efficient(ucrows *ucRows, const ldata *lData, const int *order);
void init_ucrows(ucrows *ucRows, idx_t ngcols, idx_t nlcols, idx_t *lcols, int nprocs);
void free_ucrows(ucrows *ucRows);

#endif /* end of include guard UCROWS_H */
/*
 * =====================================================================================
 *
 *       Filename:  ucrows.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  11/19/2021 06:09:49 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */

