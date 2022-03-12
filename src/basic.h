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

#ifndef BASIC_H
#define BASIC_H
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>

#define idxsize 32
#define valsize 32
#if idxsize == 32

#define idx_t uint32_t
#define IDX_T_MAX UINT32_MAX 
#define MPI_IDX_T MPI_UINT32_T

#elif idxsize == 64

#define idx_t uint64_t
#define IDX_T_MAX UINT64_MAX 
#define MPI_IDX_T MPI_UINT64_T
#endif

#define real_t double
#define MPI_REAL_T MPI_DOUBLE
#define GA_SIZE 1000000000
extern idx_t GA[ ];
typedef struct _triplet{
    idx_t row;
    idx_t col;
    real_t val;
} triplet;
typedef struct _tmr_t{
    struct timespec ts_beg;
    struct timespec ts_end;
    double elapsed;
} tmr_t;

typedef struct _node{
        idx_t gidx;
        struct _node *next;
} node;
void na_log(FILE *fp, const char* format, ...);
#endif /* end of include guard: BASIC_H */
