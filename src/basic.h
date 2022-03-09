/*
 * =====================================================================================
 *
 *       Filename:  basic.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  04-09-2020 13:38:46
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  YOUR NAME (), 
 *   Organization:  
 *
 * =====================================================================================
 */
#ifndef BASIC_H
#define BASIC_H
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <inttypes.h>

#define idxsize 64
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
FILE *dbgfp;
char dbg_fn[1024];
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
