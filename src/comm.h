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


#ifndef COMM_H_YLPYEXAA
#define COMM_H_YLPYEXAA

#include "def.h"
#include <mpi.h>
#include "ucrows.h"
#include "sschedule.h"

/*! \enum COMM_TYPE
 *
 *  Detailed description
 */
enum COMM_TYPE { NAIIVE, AFTER_UPDATE, BEFORE_NEEDED, SMART };

typedef struct _comm
{
    int nStms;
    int nprocs;
    int myrank;
    int comm_type;
    int use_fixed_length_cb;
    const int *partvec; /* column to column strip assignment */
    int commUnitSize; /* rank, f */
    idx_t *nsend;
    int **sendList; /* send and recv lists per stratum */
    idx_t *nrecv;
    int **recvList;
    idx_t **xsendinds;
    idx_t **xrecvinds; /* send and recv cnts per stratum */
    idx_t **sendinds;
    idx_t **recvinds;
    real_t *sendbuff;
    real_t *recvbuff;
    int *tags;
    MPI_Request *reqst;
    MPI_Status *stts;
    
    ucrows *ucRows;


    /* for delayed waits */
    idx_t *nrecvR; // number of irecv requests per SE
    idx_t **recvSizeR; //recvSizes (for requests) per SE 
    idx_t *nrecvA; // number of actual recvs (wait) per SE
    real_t **recvbuffDW;
    int **recvListA;
    int **recvListR;
    int **recvgtlmapA;
    MPI_Request **reqstA;
    MPI_Request ***reqstR;
    int *t_r_inds;

    int epochID;
/* aux buffers */
    idx_t *comm_L1_auxBuff1;
    idx_t *comm_L1_auxBuff2;
    idx_t **comm_L1_2DauxBuff1;
    idx_t **comm_L1_2DauxBuff2;
    idx_t *comm_L2_auxBuff1;
    idx_t *comm_L2_auxBuff2;
    idx_t **comm_L2_2DauxBuff1;
    idx_t **comm_L2_2DauxBuff2;

} Comm;

void init_comm(Comm *cm);
void free_comm(Comm *cm);
void setup_comm(Comm *cm, const ldata *lData, const sschedule *ss, const int *partvec, const int comm_type, const int use_fixed_length_cb);
void prepare_comm(Comm *cm, const ldata *lData, sschedule *ss);
void communicate_qmat_rows(real_t *qmat, const ldata *lData, const Comm *cm, const sschedule *ss, int stratumID);
#endif /* end of include guard: COMM_H_YLPYEXAA */


