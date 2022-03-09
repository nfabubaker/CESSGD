/*
 * =====================================================================================
 *
 *       Filename:  def.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  03-09-2020 12:37:02
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef DEF_H
#define DEF_H
#include "basic.h"
#include "util.h"

#ifdef TAKE_TIMES

/* extern double setuptime;
 * extern double sgdtime;
 * extern double commtime;
 * extern double lsgdtime;
 */

extern tmr_t setuptime;
extern tmr_t sgdtime;
extern tmr_t preptime;
extern tmr_t commtime;
extern tmr_t lsgdtime;
#endif

typedef struct _genst{
    int nprocs;
    int myrank;
    int f;
    int *fVals;
    int use_pfile;
    int comm_type;
    int use_randColDist;
    int sschedule_type;
    int niter;
    int *partvec;
    double lambda;
    double eps;
    idx_t gnnz;
    idx_t nnz;
    char mtxFN[1024];
    char pvecFN[1024];
} genst;

#endif /* end of include guard: DEF_H */
