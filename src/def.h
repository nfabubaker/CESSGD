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
