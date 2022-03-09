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




#include <mpi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "basic.h"
#include "comm.h"
#include "def.h"
#include "sgd.h"
#include "sschedule.h"
#include "ucrows.h"
#include "util.h"
#include <assert.h>

void communicate_qmat_rows_naive(real_t *qmat, const Comm *cm, const sschedule *ss , const int stratumID, const ldata *lData);
void communicate_updated_qmat_rows(real_t *qmat, const Comm *cm, const int stratumID);
void SM_S_afterUpdate(Comm *cm, const ldata *lData, const sschedule *ss,  const int *col_update_order){
    idx_t i, j, oidx, tmpcnt, pid, nsendto, *xsendinds = NULL, *sendinds = NULL, *tcnts = NULL, *gtlmap = NULL;
    int *sendList = NULL;
    tcnts = cm->comm_L2_auxBuff1; 
    gtlmap = cm->comm_L2_auxBuff2;
    /* for each SE */
    for (i = 0; i < cm->nprocs; ++i) {
        setIDXTArrVal(gtlmap, cm->nprocs, -1);
        setIDXTArrZero(tcnts, cm->nprocs); 
        nsendto = 0;
        oidx = ss->sorder[i]; //which Column stripe I will be updating at SE i 
        /* loop over each stratum's local columns and determine which processors update them first*/
        for (j = lData->xlcols[oidx]; j < lData->xlcols[oidx+1]; ++j) {
            pid = col_update_order[lData->lcols[j]];
            if (pid != -1) { /* if someone updates the column */
                if(tcnts[pid] == 0)
                    nsendto++;
                tcnts[pid]++; 
            }
        }

        cm->nsend[i] = nsendto;
        cm->xsendinds[i] = realloc(cm->xsendinds[i] , sizeof(*cm->xsendinds[i]) * (nsendto+2));
        xsendinds = cm->xsendinds[i];
        setIDXTArrZero(xsendinds, nsendto+2);
        if (nsendto > 0) {
            cm->sendList[i]= realloc(cm->sendList[i], sizeof(*cm->sendList[i]) * nsendto);
            sendList = cm->sendList[i];
            //cm->xsendinds[i] = calloc((nsendto+2), sizeof(*cm->xsendinds[i]));
            /* Fill the sendList and xsendinds arrays*/
            tmpcnt = 0; 
            for (j = 0; j < cm->nprocs; ++j) {
                if(tcnts[j] > 0){
                    sendList[tmpcnt] = j;
                    xsendinds[tmpcnt+2] = tcnts[j];
                    gtlmap[j] = tmpcnt;
                    tmpcnt++;
                }
            }
            for (j = 2; j < nsendto+2; ++j) {
                xsendinds[j] += xsendinds[j-1];
            }
            cm->sendinds[i] = realloc(cm->sendinds[i] , sizeof(*cm->sendinds[i]) * xsendinds[nsendto+1]);
            sendinds = cm->sendinds[i];
            /* fill the local indices to be sent to each processor */
            for (j = lData->xlcols[oidx]; j < lData->xlcols[oidx+1]; ++j) {
                pid = col_update_order[lData->lcols[j]];
                if (pid != -1) {
                    assert(gtlmap[pid] != -1);
                    sendinds[ xsendinds[ gtlmap[pid] + 1 ]++ ] = j; 
                }
            }
            assert(xsendinds[nsendto]==xsendinds[nsendto+1]);
            /* reset temp arrays */
        }
    }
}

void SM_R_afterUpdate(Comm *cm, const ldata *lData, const sschedule *ss,const  int *partvec, const idx_t *recvcnts, idx_t **trecvbuf){
    idx_t i, j, *tcnts, *ttcnts, **recvgtlmap;
    int nprocs = cm->nprocs;
    /* count processors to recv from at each sub-epoch */
    tcnts = cm->nrecv;
    ttcnts = cm->comm_L2_auxBuff1;
    setIDXTArrZero(tcnts, nprocs);
    setIDXTArrZero(ttcnts, nprocs);
    recvgtlmap = cm->comm_L2_2DauxBuff1; 
    int sidx;
    for (i = 0; i < nprocs; ++i) {
        if (recvcnts[i] > 0 && i != cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                /* get the recv SE order according to sending processor's stratum order*/
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                if(tcnts[sidx] == ttcnts[sidx])
                    tcnts[sidx]++;
            }
            for (j = 0; j < nprocs; ++j) {
                ttcnts[j] = tcnts[j];
            }
        }
    }
    /* create recvList and xrecvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        cm->xrecvinds[i] = realloc(cm->xrecvinds[i] ,(tcnts[i]+2) * sizeof(*cm->xrecvinds[i]));
        setIDXTArrZero(cm->xrecvinds[i], tcnts[i]+2);
        cm->recvList[i] = realloc(cm->recvList[i] , (tcnts[i]) * sizeof(*cm->recvList[i]));

        ttcnts[i] = tcnts[i] = 0;
    }
    /*count # inds to be recvd per stratum per processor  */
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] && i != cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                if(tcnts[sidx] == ttcnts[sidx]){
                    cm->recvList[sidx][tcnts[sidx]] = i;
                    recvgtlmap[sidx][i] = tcnts[sidx]++;
                }
                cm->xrecvinds[sidx][recvgtlmap[sidx][i]+2]++;
            }
            for (j = 0; j < nprocs; ++j) {
                ttcnts[j] = tcnts[j];
            }
        }
    }
    /* prefix sum & allocate recvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        for (j = 2; j < cm->nrecv[i]+2; ++j) {
            cm->xrecvinds[i][j] += cm->xrecvinds[i][j-1];
        }
        cm->recvinds[i] = realloc(cm->recvinds[i] , sizeof(*cm->recvinds[i]) * cm->xrecvinds[i][cm->nrecv[i]+1]);
    }
    /* fille recvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] > 0 && i!= cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                assert(sidx < nprocs);
                assert(lData->gtlcolmap[trecvbuf[i][j]] < lData->nlcols);
                assert(recvgtlmap[sidx][i] <= cm->nrecv[sidx]);
                cm->recvinds[sidx][cm->xrecvinds[sidx][recvgtlmap[sidx][i]+1]++] = lData->gtlcolmap[trecvbuf[i][j]];	
            }
        }
    }
}

void SM_S_beforeNeeded(Comm *cm, const ldata *lData, const sschedule *ss,const  int *col_update_order)
{
    idx_t i, j, oidx, tmpcnt, pid, nsendto, *xsendinds, *sendinds, *tcnts, *gtlmap;
    int *sendList;
    tcnts = realloc(tcnts , cm->nprocs * sizeof(*tcnts));
    gtlmap = realloc(gtlmap , cm->nprocs * sizeof(*gtlmap));
    cm->nsend = realloc(cm->nsend , cm->nprocs * sizeof(*cm->nsend));
    /* for each SE */
    for (i = 0; i < cm->nprocs; ++i) {
        setIDXTArrVal(gtlmap, cm->nprocs, -1);
        setIDXTArrZero(tcnts, cm->nprocs); 
        nsendto = 0;
        oidx = ss->sorder[i]; //which Column stripe I will be updating at SE i 
        /* loop over each stratum's local columns and determine which processors update them first*/
        for (j = lData->xlcols[oidx]; j < lData->xlcols[oidx+1]; ++j) {
            pid = col_update_order[lData->lcols[j]];
            if (pid != -1) { /* if someone updates the column */
                if(tcnts[pid] == 0)
                    nsendto++;
                tcnts[pid]++; 
            }
        }
        cm->nsend[i] = nsendto;
        cm->xsendinds[i] = realloc(cm->xsendinds[i] , sizeof(*cm->xsendinds[i]) * (nsendto+2));
        xsendinds = cm->xsendinds[i];
        setIDXTArrZero(xsendinds, nsendto+2);
        if (nsendto > 0) {
            cm->sendList[i]= realloc(cm->sendList[i], sizeof(*cm->sendList[i]) * nsendto);
            sendList = cm->sendList[i];
            //cm->xsendinds[i] = calloc((nsendto+2), sizeof(*cm->xsendinds[i]));
            /* Fill the sendList and xsendinds arrays*/
            tmpcnt = 0; 
            for (j = 0; j < cm->nprocs; ++j) {
                if(tcnts[j] > 0){
                    sendList[tmpcnt] = j;
                    xsendinds[tmpcnt+2] = tcnts[j];
                    gtlmap[j] = tmpcnt;
                    tmpcnt++;
                }
            }
            for (j = 2; j < nsendto+2; ++j) {
                xsendinds[j] += xsendinds[j-1];
            }
            cm->sendinds[i] = realloc(cm->sendinds[i] , sizeof(*cm->sendinds[i]) * xsendinds[nsendto+1]);
            sendinds = cm->sendinds[i];
            /* fill the local indices to be sent to each processor */
            for (j = lData->xlcols[oidx]; j < lData->xlcols[oidx+1]; ++j) {
                pid = col_update_order[lData->lcols[j]];
                if (pid != -1) {
                    assert(gtlmap[pid] != -1);
                    sendinds[ xsendinds[ gtlmap[pid] + 1 ]++ ] = j; 
                }
            }
            assert(xsendinds[nsendto]==xsendinds[nsendto+1]);
            /* reset temp arrays */
        }
    }
    /* cleanup */
    free(tcnts); free(gtlmap);
}

void SM_R_beforeNeeded(Comm *cm, const ldata *lData, const sschedule *ss, const int *partvec, const idx_t *recvcnts, idx_t **trecvbuf){
    idx_t i, j, *tcnts, *ttcntsR, *ttcntsA, **recvgtlmapR;
    int nprocs = cm->nprocs;
    cm->recvListR = realloc(cm->recvListR , sizeof(*cm->recvListR) * cm->nprocs);
    cm->recvListA = realloc(cm->recvListA , sizeof(*cm->recvListA) * cm->nprocs);
    /* count processors to recv from at each sub-epoch */
    cm->recvSizeR = realloc(cm->recvSizeR , sizeof(*cm->recvSizeR) * nprocs);
    cm->nrecvR = calloc(nprocs, sizeof(*cm->nrecvR));
    cm->nrecvA = calloc(nprocs, sizeof(*cm->nrecvA));
    cm->reqstA = realloc(cm->reqstA , nprocs * sizeof(*cm->reqstA));
    cm->reqstR = realloc(cm->reqstR , nprocs * sizeof(*cm->reqstR));
    tcnts = cm->nrecvR;
    ttcntsR = calloc(nprocs, sizeof(*ttcntsR));
    ttcntsA = calloc(nprocs, sizeof(*ttcntsA));
    cm->recvgtlmapA = realloc(cm->recvgtlmapA , sizeof(*cm->recvgtlmapA) * nprocs);
    recvgtlmapR = realloc(recvgtlmapR , sizeof(*recvgtlmapR) * nprocs);
    cm->recvbuffDW = realloc(cm->recvbuffDW , sizeof(*cm->recvbuffDW) * nprocs);
    int sidx, msidx;
    for (i = 0; i < nprocs; ++i) {
        cm->recvgtlmapA[i] = realloc(cm->recvgtlmapA[i] , sizeof(*cm->recvgtlmapA[i])*nprocs);
        recvgtlmapR[i] = realloc(recvgtlmapR[i] , sizeof(*recvgtlmapR[i])*nprocs);
        if (recvcnts[i] > 0 && i != cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                /* get the recv SE order according to sending processor's stratum order*/
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                /* when do I need this column ? (before which SE ?) */
                msidx = (get_se(cm->myrank, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed) - 1 + nprocs ) % nprocs;
                tcnts[sidx] += (tcnts[sidx] == ttcntsR[sidx]? 1:0);
                cm->nrecvA[msidx] += (cm->nrecvA[msidx] == ttcntsA[msidx] ? 1:0); 
            }
            for (j = 0; j < nprocs; ++j) {
                ttcntsR[j] = tcnts[j];
                ttcntsA[j] = cm->nrecvA[j];
            }
        }
    }
    /* create recvList and xrecvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        cm->xrecvinds[i] = calloc((cm->nrecvA[i]+2) , sizeof(*cm->xrecvinds[i]));
        cm->recvListA[i] = realloc(cm->recvListA[i] , (cm->nrecvA[i]) * sizeof(*cm->recvListA[i]));
        cm->recvListR[i] = realloc(cm->recvListR[i] , (tcnts[i]) * sizeof(*cm->recvListR[i]));
        cm->recvSizeR[i] = calloc((tcnts[i]), sizeof(*cm->recvSizeR[i]));
        cm->reqstR[i] = realloc(cm->reqstR[i] , tcnts[i] * sizeof(*cm->reqstR[i]));
        cm->reqstA[i] = realloc(cm->reqstA[i] , cm->nrecvA[i] * sizeof(*cm->reqstA[i]));
        ttcntsR[i] = tcnts[i] = 0;
        ttcntsA[i] = cm->nrecvA[i] = 0;
    }
    /*count # inds to be recvd per stratum per processor  */
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] && i != cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                msidx = (get_se(cm->myrank, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed) - 1 + nprocs ) % nprocs;
                if(tcnts[sidx] == ttcntsR[sidx]){
                    cm->recvListR[sidx][tcnts[sidx]] = i;
                    recvgtlmapR[sidx][i] = tcnts[sidx]++;
                }
                if(cm->nrecvA[msidx] == ttcntsA[msidx]){
                    cm->recvListA[msidx][cm->nrecvA[msidx]] = i;
                    cm->recvgtlmapA[msidx][i] = cm->nrecvA[msidx]++;
                    //cm->reqstA[msidx][cm->recvgtlmapA[msidx][i]] = &cm->reqstR[sidx][recvgtlmapR[sidx][i]]; //map requests in wait stage to the original stage
                    cm->reqstR[sidx][recvgtlmapR[sidx][i]] = &cm->reqstA[msidx][cm->recvgtlmapA[msidx][i]] ; //map requests in wait stage to the original stage
                }
                cm->xrecvinds[msidx][cm->recvgtlmapA[msidx][i]+2]++;
                cm->recvSizeR[sidx][recvgtlmapR[sidx][i]]++;
            }
            for (j = 0; j < nprocs; ++j) {
                ttcntsR[j] = tcnts[j];
                ttcntsA[j] = cm->nrecvA[j];
            }
        }
    }
    /* prefix sum & allocate recvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        for (j = 2; j < cm->nrecvA[i]+2; ++j) {
            cm->xrecvinds[i][j] += cm->xrecvinds[i][j-1];
        }
        cm->recvinds[i] = realloc(cm->recvinds[i] , sizeof(*cm->recvinds[i]) * cm->xrecvinds[i][cm->nrecvA[i]+1]);
        cm->recvbuffDW[i] = realloc(cm->recvbuffDW[i] , sizeof(*cm->recvbuffDW[i]) * cm->xrecvinds[i][cm->nrecvA[i]+1]*cm->commUnitSize);
    }
    /* fille recvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] > 0 && i!= cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                //sidx = get_se(cm->myrank, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                sidx = (get_se(cm->myrank, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed) - 1 + nprocs ) % nprocs;
                assert(sidx < nprocs);
                assert(lData->gtlcolmap[trecvbuf[i][j]] < lData->nlcols);
                assert(cm->recvgtlmapA[sidx][i] <= cm->nrecvA[sidx]);
                cm->recvinds[sidx][cm->xrecvinds[sidx][cm->recvgtlmapA[sidx][i]+1]++] = lData->gtlcolmap[trecvbuf[i][j]];	
            }
        }
    }
    /* cleanup */
    for (i = 0; i < cm->nprocs; ++i) {
        free(recvgtlmapR[i]);
        recvgtlmapR[i] = NULL;
    }
    free(ttcntsA); free(ttcntsR); free(recvgtlmapR);
}

int smart_schedule_msg_send(sschedule const *ss, int px, int py, int SE, int K){
    int sendATSE, dist, mid;
    dist = (K+(get_se(py, K, 1, ss->seed) - get_se(px, K, 1, ss->seed)) )% K;
    mid = SE/dist;
    /* which SE px updated the first CB of this message?  */
    int tse = mid * dist;
    /* get the first CB id of this msg */
    int betaB = get_cbidx(px, K, tse, ss->seed);
    /* when does py update this CB ? */
    sendATSE = get_se(py, K, betaB, ss->seed);
    /* we send in the prev. step */
    sendATSE = (K+(sendATSE - 1)) %K;
    return sendATSE;
}

void SM_S_Smart(Comm *cm, const ldata *lData, const sschedule *ss, const int *col_update_order){
    idx_t  j, tmpcnt, nsendto, *xsendinds, **tcnts, **gtlmap;
    int dist, mID, sendATSE, *sendList, pid, oidx, i;
    tcnts = cm->comm_L2_2DauxBuff1;
    gtlmap = cm->comm_L2_2DauxBuff2;
    setIDXTArrZero(cm->nsend, cm->nprocs);

    /* for each SE */
    for (i = 0; i < cm->nprocs; ++i) {
        setIDXTArrZero(tcnts[i], cm->nprocs); 
    }
    for (i = 0; i < cm->nprocs; ++i) {
        nsendto = 0;
        oidx = ss->sorder[i]; //which Column stripe I will be updating at SE i 
        /* loop over each stratum's local columns and determine which processors update them first*/
        for (j = lData->xlcols[oidx]; j < lData->xlcols[oidx+1]; ++j) {
            pid = col_update_order[lData->lcols[j]];
            if (pid != -1) { /* if someone updates the column */
                sendATSE = smart_schedule_msg_send(ss, cm->myrank, pid, i, cm->nStms);
                if(tcnts[sendATSE][pid] == 0)
                    cm->nsend[sendATSE]++;
                tcnts[sendATSE][pid]++; 
            }
        }
    }
    for (i = 0; i < cm->nprocs; ++i) { /* for each SE */
        setIDXTArrVal(gtlmap[i], cm->nprocs, IDX_T_MAX);
        nsendto = cm->nsend[i];
        cm->xsendinds[i] = realloc(cm->xsendinds[i] , sizeof(*cm->xsendinds[i]) * (nsendto+2));
        xsendinds = cm->xsendinds[i];
        setIDXTArrZero(xsendinds, nsendto+2);
        if (nsendto > 0) {
            cm->sendList[i]= realloc(cm->sendList[i], sizeof(*cm->sendList[i]) * nsendto);
            sendList = cm->sendList[i];
            //cm->xsendinds[i] = calloc((nsendto+2), sizeof(*cm->xsendinds[i]));
            /* Fill the sendList and xsendinds arrays*/
            tmpcnt = 0; 
            for (j = 0; j < cm->nprocs; ++j) {
                if(tcnts[i][j] > 0){
                    sendList[tmpcnt] = j;
                    xsendinds[tmpcnt+2] = tcnts[i][j];
                    gtlmap[i][j] = tmpcnt;
                    tmpcnt++;
                }
            }
            for (j = 2; j < nsendto+2; ++j) {
                xsendinds[j] += xsendinds[j-1];
            }
            cm->sendinds[i] = realloc(cm->sendinds[i] , sizeof(*cm->sendinds[i]) * xsendinds[nsendto+1]);
        }
    }
    //       sendinds = cm->sendinds[i];
    for (i = 0; i < cm->nprocs; ++i) {
        /* fill the local indices to be sent to each processor */
        oidx = ss->sorder[i]; //which Column stripe I will be updating at SE i 
        for (j = lData->xlcols[oidx]; j < lData->xlcols[oidx+1]; ++j) {
            pid = col_update_order[lData->lcols[j]];
            if (pid != -1) {
                sendATSE = smart_schedule_msg_send(ss, cm->myrank, pid, i, cm->nStms);
                assert(gtlmap[sendATSE][pid] != IDX_T_MAX);
                cm->sendinds[sendATSE][cm->xsendinds[sendATSE][ gtlmap[sendATSE][pid] + 1 ]++ ] = j; 
            }
        }
    }
}

void SM_R_Smart(Comm *cm, const ldata *lData, const sschedule *ss, const  int *partvec,const  idx_t *recvcnts, idx_t **trecvbuf){
    idx_t i, j, *tcnts, *ttcnts, **recvgtlmap;
    int nprocs = cm->nprocs, dist, mID, recvATSE;
    /* count processors to recv from at each sub-epoch */
    setIDXTArrZero(cm->nrecv, nprocs);
    tcnts = cm->nrecv;
    ttcnts = cm->comm_L2_auxBuff1;
    setIDXTArrZero(ttcnts, nprocs);
    recvgtlmap = cm->comm_L2_2DauxBuff1;
    int sidx;
    for (i = 0; i < nprocs; ++i) { /* for each processor */
        if (recvcnts[i] > 0 && i != cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                /* get the recv SE order according to sending processor's stratum order*/
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                //dist = (i - cm->myrank + cm->nprocs) % cm->nprocs;
                //mID = sidx / dist; 
                recvATSE = smart_schedule_msg_send(ss, i, cm->myrank, sidx, cm->nStms);
                if(tcnts[recvATSE] == ttcnts[recvATSE])
                    tcnts[recvATSE]++;
            }
            for (j = 0; j < nprocs; ++j) {
                ttcnts[j] = tcnts[j];
            }
        }
    }
    /* create recvList and xrecvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        cm->xrecvinds[i] = realloc( cm->xrecvinds[i] ,(tcnts[i]+2) * sizeof(*cm->xrecvinds[i]));
        setIDXTArrZero(cm->xrecvinds[i], tcnts[i] +2 );
        cm->recvList[i] = realloc(cm->recvList[i] , (tcnts[i]) * sizeof(*cm->recvList[i]));
        ttcnts[i] = tcnts[i] = 0;
    }
    /*count # inds to be recvd per stratum per processor  */
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] > 0 && i != cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                /*                 dist = (i - cm->myrank + cm->nprocs) % cm->nprocs;
                 *                 mID = sidx / dist;
                 *                 recvATSE = ((mID*dist)+(dist-1)) % cm->nprocs;
                 */
                recvATSE = smart_schedule_msg_send(ss, i, cm->myrank, sidx, cm->nStms);
                if(tcnts[recvATSE] == ttcnts[recvATSE]){
                    cm->recvList[recvATSE][tcnts[recvATSE]] = i;
                    recvgtlmap[recvATSE][i] = tcnts[recvATSE]++;
                }
                cm->xrecvinds[recvATSE][recvgtlmap[recvATSE][i]+2]++;
            }
            for (j = 0; j < nprocs; ++j) {
                ttcnts[j] = tcnts[j];
            }
        }
    }
    /* prefix sum & allocate recvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        for (j = 2; j < cm->nrecv[i]+2; ++j) {
            cm->xrecvinds[i][j] += cm->xrecvinds[i][j-1];
        }
        cm->recvinds[i] = realloc(cm->recvinds[i] , sizeof(*cm->recvinds[i]) * cm->xrecvinds[i][cm->nrecv[i]+1]);
    }
    /* fille recvinds arrays */
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] > 0 && i!= cm->myrank){
            for (j = 0; j < recvcnts[i]; ++j) {
                sidx = get_se(i, cm->nprocs, partvec[trecvbuf[i][j]], ss->seed);
                recvATSE = smart_schedule_msg_send(ss, i, cm->myrank, sidx, cm->nStms);
                /*                 dist = (i - cm->myrank + cm->nprocs) % cm->nprocs;
                 *                 mID = sidx / dist;
                 *                 recvATSE = ((mID*dist)+(dist-1)) % cm->nprocs;
                 */
                assert(recvATSE < nprocs);
                assert(lData->gtlcolmap[trecvbuf[i][j]] < lData->nlcols);
                assert(recvgtlmap[recvATSE][i] <= cm->nrecv[recvATSE]);
                cm->recvinds[recvATSE][cm->xrecvinds[recvATSE][recvgtlmap[recvATSE][i]+1]++] = lData->gtlcolmap[trecvbuf[i][j]];	
            }
        }
    }
}
void schedule_send_messages(Comm *cm, const ldata *lData, const sschedule *ss, const int *col_update_order){
    switch (cm->comm_type) {
        case AFTER_UPDATE:
        case BEFORE_NEEDED:
            SM_S_afterUpdate(cm, lData,ss, col_update_order);
            break;
        case SMART:
            SM_S_Smart(cm, lData,ss, col_update_order);
        default:
            break;

    }
}

void schedule_recv_messages(Comm *cm, const ldata *lData, const sschedule *ss, const int *partvec, const idx_t *recvcnts,  idx_t **trecvbuf){
    switch (cm->comm_type) {
        case AFTER_UPDATE:
            SM_R_afterUpdate(cm, lData,ss, partvec, recvcnts, trecvbuf);
            break;
        case BEFORE_NEEDED:
            SM_R_beforeNeeded(cm, lData, ss, partvec, recvcnts, trecvbuf);
            break;
        case SMART:
            SM_R_Smart(cm, lData,ss, partvec, recvcnts, trecvbuf);
            break;
        default:
            break;

    }
}

void update_comm(Comm *cm, const ldata *lData, const sschedule *ss, const int *partvec, const int *col_update_order){
    int i, j, k, nprocs; 
    idx_t *tsendbuf, **trecvbuf, *sendcnts, *recvcnts;
    nprocs = cm->nprocs;

    /* schedule_messages */
    schedule_send_messages(cm, lData, ss, col_update_order);

    /* determine total send cnts per processor */
    sendcnts = cm->comm_L1_auxBuff1;
    setIDXTArrZero(sendcnts, nprocs);
    recvcnts = cm->comm_L1_auxBuff2;
    setIDXTArrVal(recvcnts, cm->nprocs, 0);
    idx_t tcnt=0, ttotcnt=0;
    for (i = 0; i < nprocs; ++i) { /* for each SE */
        for (j = 0; j < cm->nsend[i]; ++j) { /* for each processor in sendList */
            tcnt = (cm->xsendinds[i][j+1] - cm->xsendinds[i][j]);  
            ttotcnt += tcnt;
            sendcnts[cm->sendList[i][j]] +=  tcnt;
        }   
    }
    /* send counts to each corresponding processor */
    MPI_Alltoall(sendcnts, 1, MPI_IDX_T, recvcnts, 1, MPI_IDX_T, MPI_COMM_WORLD);
    idx_t max_send=0, max_recv=0;
    for (i = 0; i < nprocs; ++i) {
        if (sendcnts[i] > max_send)
            max_send = sendcnts[i];
        if(recvcnts[i] > max_recv)
            max_recv = recvcnts[i];
    }

    tsendbuf = NULL;
    trecvbuf = NULL;
    tsendbuf = realloc(tsendbuf , max_send*sizeof(*tsendbuf));
    trecvbuf = realloc(trecvbuf , sizeof(*trecvbuf) * nprocs);
    int tnrecv = 0;
    for (i = 0; i < nprocs; ++i) {
#ifdef NA_DBG_L2
        na_log(dbgfp, "\t\trecvcnt[%d]=%d\n", i, recvcnts[i]);
#endif
        trecvbuf[i] = NULL;
        if(recvcnts[i] > 0){
            trecvbuf[i] = realloc(trecvbuf[i] , sizeof(*trecvbuf[i]) * recvcnts[i]);
            tnrecv++;
        }
    }

    MPI_Request req[tnrecv];
    tnrecv = 0;
    for (i = 0; i < nprocs; ++i) {
        if(recvcnts[i] > 0){
            MPI_Irecv(trecvbuf[i], recvcnts[i], MPI_IDX_T, i, 7, MPI_COMM_WORLD, &req[tnrecv++]);
        }
    }
    int tpidx;
    idx_t tmpvar,tmpcnt,  *tptr;
    for (i = 0; i < nprocs; ++i) {
        tmpcnt = 0;
        if (sendcnts[i] > 0) {
            /* prepare tsendbuf of indices to be sent to processor i */
            tptr = tsendbuf;
            for (j = 0; j < nprocs; ++j) { /* for each stratum */
                /* first, determine the local idx of processor i in sendlist of stratum j */
                tpidx = -1;
                for(k=0; k < cm->nsend[j]; k++){
                    if(cm->sendList[j][k] == i){
                        tpidx = k;
                        break;
                    }
                }
                if(tpidx == -1) continue; //not in this stratum
                for (tmpvar=0, k = cm->xsendinds[j][tpidx]; k < cm->xsendinds[j][tpidx+1]; ++k, ++tmpvar) {
                    assert(cm->sendinds[j][k] < lData->nlcols && cm->sendinds[j][k] >= 0);
                    *(tptr++) = lData->lcols[cm->sendinds[j][k]]; /* assign global col idxs to the buffer */
                }
                tmpcnt+= tmpvar;
            }

#ifdef NA_DBG_L2
            na_log(dbgfp, "\tbefore send, tmpcnt=%d - sendcnts[%d]=%d\n", tmpcnt, i,sendcnts[i]);
#endif
            assert(tmpcnt == sendcnts[i]); /* make sure everything is fine so far */
            MPI_Send(tsendbuf, tmpcnt, MPI_IDX_T, i, 7, MPI_COMM_WORLD);
        }
    }
    /* TODO: wait for irecvs and process them, remember the inds are global <12-10-20, yourname> */ 
    MPI_Waitall(tnrecv, req, MPI_STATUSES_IGNORE);

    /* #ifdef NA_DBG
     *     {
     *         volatile int tt = 0;
     *         printf("PID %d on %d ready for attach\n",gs->myrank,  getpid());
     *         fflush(stdout);
     *         while (0 == tt)
     *             sleep(5);
     *     }
     * 
     * #endif
     */
    schedule_recv_messages(cm, lData, ss, partvec, recvcnts, trecvbuf);
    /* allocate send and recv buffers */

    idx_t mxsnd = 0, mxrecv = 0, mxnsnd =0, mxnrecv=0;
    if (cm->comm_type == BEFORE_NEEDED) {
        for (i = 0; i < nprocs; ++i) {
            assert(cm->xrecvinds[i][cm->nrecvA[i]] == cm->xrecvinds[i][cm->nrecvA[i]+1]);
            if(cm->xsendinds[i][cm->nsend[i]] > mxsnd )
                mxsnd = cm->xsendinds[i][cm->nsend[i]];
            if(cm->xrecvinds[i][cm->nrecvA[i]] > mxrecv)
                mxrecv = cm->xrecvinds[i][cm->nrecvA[i]];
            mxnsnd = (cm->nsend[i] > mxnsnd) ? cm->nsend[i] : mxnsnd;
            mxnrecv = (cm->nrecvA[i] > mxnrecv) ? cm->nrecvA[i] : mxnrecv;
        }
        cm->t_r_inds = realloc(cm->t_r_inds , sizeof(*cm->t_r_inds) * mxnrecv);
    }
    else{    
        for (i = 0; i < nprocs; ++i) {
            assert(cm->xrecvinds[i][cm->nrecv[i]] == cm->xrecvinds[i][cm->nrecv[i]+1]);
            if(cm->xsendinds[i][cm->nsend[i]] > mxsnd )
                mxsnd = cm->xsendinds[i][cm->nsend[i]];
            if(cm->xrecvinds[i][cm->nrecv[i]] > mxrecv)
                mxrecv = cm->xrecvinds[i][cm->nrecv[i]];
            mxnsnd = (cm->nsend[i] > mxnsnd) ? cm->nsend[i] : mxnsnd;
            mxnrecv = (cm->nrecv[i] > mxnrecv) ? cm->nrecv[i] : mxnrecv;
        }

        cm->recvbuff = realloc(cm->recvbuff , sizeof(*cm->recvbuff) *mxrecv * cm->commUnitSize);
    }


    cm->sendbuff = realloc(cm->sendbuff , sizeof(*cm->sendbuff)*mxsnd * cm->commUnitSize);
    /* cleanup */
    for (i = 0; i < cm->nprocs; ++i) {
        if(trecvbuf[i] != NULL)
            free(trecvbuf[i]);
    }
    free(trecvbuf);
    free(tsendbuf);
}

void prepare_comm(Comm *cm, const ldata *lData, sschedule *ss){
    switch (cm->comm_type) {
        case NAIIVE:
            /* do nothing */
            update_sschedule(ss, lData->nprocs);
            break;
        case BEFORE_NEEDED:
        case SMART:
        case AFTER_UPDATE:
            /* update_sschedule and update_comm should always be called together */
            if(update_sschedule(ss, lData->nprocs)){
                get_shared_cols_efficient(cm->ucRows, lData, ss->order);
                update_comm(cm, lData, ss, cm->partvec, cm->ucRows->col_update_order);
            }
            else 
                return;
            break;
    }
}

void setup_naiv_comm(Comm *cm, const ldata *lData){
    //cm->sendbuff = malloc(((gs->ngcols/gs->nprocs) + 1)*gs->f *sizeof(cm->sendbuff));
    cm->sendbuff = malloc(lData->maxColStrip * cm->commUnitSize *sizeof(cm->sendbuff));
    mat_init(cm->sendbuff, lData->maxColStrip * cm->commUnitSize, cm->commUnitSize);
    if(cm->use_fixed_length_cb == 0){
        cm->recvbuff = malloc(lData->maxColStrip * cm->commUnitSize *sizeof(cm->recvbuff));
        mat_init(cm->recvbuff, lData->maxColStrip * cm->commUnitSize, cm->commUnitSize); 
    }
}


void setup_p2p_comm(Comm *cm, const ldata *lData, const sschedule *ss){
    int i;

    cm->sendList = malloc( sizeof(*cm->sendList) * cm->nStms);
    cm->recvList = malloc( sizeof(*cm->recvList) * cm->nStms);
    cm->xsendinds = malloc( sizeof(*cm->xsendinds) * cm->nStms);
    cm->sendinds = malloc( sizeof(*cm->sendinds) * cm->nStms);
    cm->xrecvinds = malloc( sizeof(*cm->xrecvinds) * cm->nStms);
    cm->recvinds = malloc( sizeof(*cm->xrecvinds) * cm->nStms);
    cm->nsend = malloc(cm->nStms * sizeof(*cm->nsend));
    cm->nrecv = calloc(cm->nStms, sizeof(*cm->nrecv));
    cm->tags = malloc( sizeof(*cm->tags)*cm->nStms); //one tag per stratum

    cm->comm_L1_auxBuff1 = malloc(sizeof(*cm->comm_L1_auxBuff1) * cm->nStms);
    cm->comm_L1_auxBuff2 = malloc(sizeof(*cm->comm_L1_auxBuff1) * cm->nStms);
    cm->comm_L2_auxBuff1 = malloc(sizeof(*cm->comm_L1_auxBuff1) * cm->nStms);
    cm->comm_L2_auxBuff2 = malloc(sizeof(*cm->comm_L1_auxBuff1) * cm->nStms);
    cm->comm_L1_2DauxBuff1 = malloc(sizeof(*cm->comm_L1_2DauxBuff1) * cm->nStms);
    cm->comm_L1_2DauxBuff2 = malloc(sizeof(*cm->comm_L1_2DauxBuff1) * cm->nStms);
    cm->comm_L2_2DauxBuff1 = malloc(sizeof(*cm->comm_L1_2DauxBuff1) * cm->nStms);
    cm->comm_L2_2DauxBuff2 = malloc(sizeof(*cm->comm_L1_2DauxBuff1) * cm->nStms);

    for (i = 0; i < cm->nStms; ++i) {
        cm->sendList[i] = NULL;
        cm->recvList[i] = NULL;
        cm->xsendinds[i] = NULL;
        cm->xrecvinds[i] = NULL;
        cm->sendinds[i] = NULL;
        cm->recvinds[i] = NULL;
        cm->comm_L1_2DauxBuff1[i] = malloc(sizeof(*cm->comm_L1_2DauxBuff1[i]) * cm->nStms);
        cm->comm_L1_2DauxBuff2[i] = malloc(sizeof(*cm->comm_L1_2DauxBuff1[i]) * cm->nStms);
        cm->comm_L2_2DauxBuff1[i] = malloc(sizeof(*cm->comm_L1_2DauxBuff1[i]) * cm->nStms);
        cm->comm_L2_2DauxBuff2[i] = malloc(sizeof(*cm->comm_L1_2DauxBuff1[i]) * cm->nStms);
    }
    cm->sendbuff = NULL;
    cm->recvbuff = NULL;

    update_comm(cm, lData, ss, cm->partvec, cm->ucRows->col_update_order);

    cm->stts = malloc(cm->nprocs * sizeof(*cm->stts));
    cm->reqst = malloc(cm->nprocs * sizeof(*cm->reqst));
    for (i = 0; i < cm->nStms; ++i) {
        cm->tags[i] = 777+i;
    }

}

void init_comm(Comm * cm){
    cm->comm_L1_2DauxBuff1 = NULL;
    cm->comm_L1_2DauxBuff2 = NULL;
    cm->comm_L1_auxBuff1 = NULL;
    cm->comm_L1_auxBuff2 = NULL;
    cm->comm_L2_2DauxBuff1 = NULL;
    cm->comm_L2_2DauxBuff2 = NULL;
    cm->comm_L2_auxBuff1 = NULL;
    cm->nrecv = NULL;
    cm->nsend = NULL;
    cm->recvList = NULL;
    cm->sendList = NULL;
    cm->recvbuff = NULL;
    cm->sendbuff = NULL;
    cm->recvinds = NULL;
    cm->xrecvinds = NULL;
    cm->sendinds = NULL;
    cm->xsendinds = NULL;
    cm->ucRows = NULL;
}

void free_comm(Comm *cm){
    
    int i;
    if (cm->ucRows != NULL) 
        free_ucrows(cm->ucRows);
    if( cm->comm_L1_2DauxBuff1 != NULL) {
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->comm_L1_2DauxBuff1[i] != NULL )
                free( cm->comm_L1_2DauxBuff1[i] );
        }
        free(cm->comm_L1_2DauxBuff1);
    }
    if( cm->comm_L1_2DauxBuff2 != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if(  cm->comm_L1_2DauxBuff2[i] != NULL )
                free(  cm->comm_L1_2DauxBuff2[i] );
        }
        free(  cm->comm_L1_2DauxBuff2);
    }

    if( cm->comm_L1_auxBuff1 != NULL)
        free(cm->comm_L1_auxBuff1);
    if(cm->comm_L1_auxBuff2 != NULL)
        free(cm->comm_L1_auxBuff2);
    if( cm->comm_L2_2DauxBuff1 != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if(  cm->comm_L2_2DauxBuff1 [i] != NULL )
                free(  cm->comm_L2_2DauxBuff1 [ i] );
        }
        free(cm->comm_L2_2DauxBuff1);
    }

    if( cm->comm_L2_2DauxBuff2 != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if(  cm->comm_L2_2DauxBuff2 [ i] != NULL )
                free(  cm->comm_L2_2DauxBuff2 [ i ] );
        }
        free(cm->comm_L2_2DauxBuff2);
    }

    if(cm->comm_L2_auxBuff1 != NULL){
        free(cm->comm_L2_auxBuff1);
    }

    if(  cm->nrecv != NULL)
        free(cm->nrecv);
    if(  cm->nsend != NULL)
        free(cm->nsend);
    if(  cm->recvList != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->recvList[i] != NULL ){
                free( cm->recvList[i] );
                cm->recvList[i] = NULL;
            }
        }
        free(cm->recvList);
    }

    if(  cm->sendList != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->sendList[i] != NULL ){
                free( cm->sendList[i] );
                cm->sendList[i] = NULL;
            }
        }
        free(cm->sendList);
    }

    if(  cm->recvbuff != NULL)
        free(cm->recvbuff);
    if(  cm->sendbuff != NULL)
        free(cm->sendbuff);
    if(  cm->recvinds != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->recvinds[i] != NULL ){
                free( cm->recvinds[i] );
                cm->recvinds[i] = NULL;
            }
        }
        free(cm->recvinds);
    }

    if(  cm->xrecvinds != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->xrecvinds[i] != NULL ){
                free( cm->xrecvinds[i] );
                cm->xrecvinds[i] = NULL;
            }
        }
        free(cm->xrecvinds);
    }

    if(  cm->sendinds != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->sendinds[i] != NULL ){
                free( cm->sendinds[i] );
                cm->sendinds[i] = NULL;
            }
        }
        free(cm->sendinds);
    }

    if(  cm->xsendinds != NULL){
        for (i = 0; i < cm->nprocs; ++i) {
            if( cm->xsendinds[i] != NULL ){
                free( cm->xsendinds[i] );
                cm->xsendinds[i] = NULL;
            }
        }
        free(cm->xsendinds);
    }


}

void setup_comm(Comm *cm, const ldata *lData, const sschedule *ss, const int *partvec, const int comm_type, const int use_fixed_length_cm){
    int mypid;
    MPI_Comm_rank(MPI_COMM_WORLD, &mypid);
    cm->use_fixed_length_cb = use_fixed_length_cm;
    cm->comm_type = comm_type;
    cm->partvec = partvec;
    cm->myrank = mypid;
    cm->nStms = lData->nStms;
    cm->nprocs = lData->nprocs;
    cm->commUnitSize = lData->f;

    init_comm(cm);

    if(comm_type == NAIIVE){
        setup_naiv_comm(cm, lData);
        return ;
    }
    cm->ucRows = malloc(sizeof(*cm->ucRows));
    init_ucrows(cm->ucRows, lData->ngcols, lData->nlcols, lData->lcols, cm->nprocs);
    get_shared_cols_efficient(cm->ucRows, lData, ss->order);
    setup_p2p_comm(cm, lData, ss);
}


void communicate_qmat_rows_naive(real_t *qmat, const Comm *cm, const sschedule *ss, const int stratumID, const ldata *lData){

    int i,f, cidx,cidx2, dst, src, K;
    idx_t bsize;
    K = cm->nprocs;
    f = cm->commUnitSize;
    dst = (cm->myrank - 1 + cm->nprocs) % cm->nprocs;
    src = (cm->myrank + 1) % cm->nprocs;
    bsize = (lData->ngcols / cm->nprocs) + (lData->ngcols%K ? 1:0);
    real_t *tb;
    /* copy to sendbuff */
    tb = cm->sendbuff;
    cidx = ss->sorder[stratumID]; 
    for (i = lData->xlcols[cidx]; i < lData->xlcols[cidx+1] ; ++i) {
        memcpy(tb, &qmat[lData->lcols[i] * f], sizeof(*qmat)*f);
        tb += f;
    }
    if(cm->use_fixed_length_cb == 0){
        cidx2 = ss->sorder[(stratumID+1) % K];
        MPI_Sendrecv(cm->sendbuff, (lData->xlcols[cidx+1]-lData->xlcols[cidx])*f, MPI_REAL_T, dst, 777, cm->recvbuff, (lData->xlcols[cidx2+1]-lData->xlcols[cidx2])*f, MPI_REAL_T, src, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* copy back */
        tb = cm->recvbuff;
        for (i = lData->xlcols[cidx2]; i < lData->xlcols[cidx2+1] ; ++i) {
            memcpy(&qmat[lData->lcols[i] * f], tb , sizeof(*qmat)*f);
            tb += f;
        }
    }
    else{
        MPI_Sendrecv_replace(cm->sendbuff, bsize * f, MPI_REAL_T, dst, 777, src, 777, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        /* copy back */
        cidx = ss->sorder[(stratumID+1) % K];
        tb = cm->sendbuff;
        for (i = lData->xlcols[cidx]; i < lData->xlcols[cidx+1] ; ++i) {
            memcpy(&qmat[lData->lcols[i] * f], tb , sizeof(*qmat)*f);
            tb += f;
        }
    }
}

void communicate_qmat_rows(real_t *qmat, const ldata *lData, const Comm *cm, const sschedule *ss,  int stratumID){
    switch (cm->comm_type) {
        case NAIIVE:
            communicate_qmat_rows_naive(qmat, cm, ss, stratumID, lData );
            break;
        case AFTER_UPDATE:
        case SMART:
            communicate_updated_qmat_rows(qmat, cm, stratumID );
            break;
        case BEFORE_NEEDED:
            //communicate_updated_qmat_rows_dwaits(qmat, cm, stratumID, cm->commUnitSize);
            break;
    }
}
void communicate_updated_qmat_rows(real_t *qmat, const Comm *cm, const int stratumID){

    int i, j, f;
    f = cm->commUnitSize;


    for (i = 0; i < cm->nrecv[stratumID]; ++i) {
        MPI_Irecv(&cm->recvbuff[cm->xrecvinds[stratumID][i] * f], f* (cm->xrecvinds[stratumID][i+1]-cm->xrecvinds[stratumID][i]), MPI_REAL_T, cm->recvList[stratumID][i], cm->tags[stratumID], MPI_COMM_WORLD, &cm->reqst[i]);
    }   


    /* prepare data to send */
    for (i = 0; i < cm->nsend[stratumID]; ++i) {
        for (j = cm->xsendinds[stratumID][i]; j < cm->xsendinds[stratumID][i+1]; ++j) {
            memcpy(&cm->sendbuff[j*f], &qmat[cm->sendinds[stratumID][j] * f], f * sizeof(*cm->sendbuff));
        }
    }
    for (i = 0; i < cm->nsend[stratumID]; ++i) {
        MPI_Send(&cm->sendbuff[cm->xsendinds[stratumID][i]*f], f*(cm->xsendinds[stratumID][i+1]-cm->xsendinds[stratumID][i]), MPI_REAL_T, cm->sendList[stratumID][i], cm->tags[stratumID], MPI_COMM_WORLD);
    }

    /* copy recvd data */
    MPI_Waitall(cm->nrecv[stratumID], cm->reqst, cm->stts);


    for (i = 0; i < cm->nrecv[stratumID]; ++i) {
        for (j = cm->xrecvinds[stratumID][i]; j < cm->xrecvinds[stratumID][i+1]; ++j) {
            memcpy(&qmat[cm->recvinds[stratumID][j]*f], &cm->recvbuff[j*f], f * sizeof(*cm->recvbuff));
        }
    }
}
