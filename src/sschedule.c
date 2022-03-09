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


#include "sschedule.h"
#include "mpi.h"
#include <assert.h>
#include <time.h>
#include <stdlib.h>

/* get CS from pid and SE */
inline int get_oidx(int pid, int nprocs, int SE){
    return (pid+SE)%nprocs;
}
/* get SE from pid and CS */
inline int get_roidx(int pid, int nprocs, int CS){
    return (nprocs-pid+CS)%nprocs;
}
/* get CS from pid and SE */
inline int get_cbidx(int pid, int K, int SE, int seed){
    return (((pid + seed) % K)+SE)%K;
}
/* get SE from pid and CS */
inline int get_se(int pid, int K, int CS, int seed){
    return (K-((pid + seed) % K)+CS)%K;
}

void get_ring_order(int *arr, int size, int start) {
    int i;
    arr[0] = start;
    for (i = 1; i < size; ++i) {
        arr[i] = (arr[i - 1] - 1 + size) % (size);
    }
}

void init_sschedule(sschedule *ss, int K){
    int myrank, i, seed;

    ss->odist = NULL;
    ss->order = NULL;
    ss->sorder = NULL;

    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0) {
        switch (ss->schedule_type) {
            case RING_RANDOM_SEED:
                srand((unsigned)time(NULL));
                seed = rand() % K;
                break;
            case RING_FIXED_SEED:
            default:
                seed = 0;
                break;
        }
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ss->seed = seed;
    ss->order = malloc(sizeof(*ss->order) * K);
    get_ring_order(ss->order, K, myrank);
    ss->sorder = malloc(sizeof(*ss->sorder) * K);
    for (i = 0; i < K; ++i) {
        ss->sorder[i] = get_cbidx(myrank, K, i, seed); 
    }
    /* distance according to order */
    ss->odist = malloc(sizeof(*ss->odist) * K);
    for (i = 0; i < K; ++i) {
        int dist, tpidx = ss->order[i];
        ss->odist[tpidx] = i;
        dist = (K + (get_se(tpidx, K, 2, seed) - get_se(myrank, K, 2, seed))) % K;
        assert(ss->odist[ss->order[i]] == dist);
    }
}
void free_ss(sschedule *ss){
    free(ss->sorder);
    free(ss->order);
    free(ss->odist);
}

int update_sschedule(sschedule *ss, int K){
    int myrank, i, seed;
    if (ss->schedule_type == RING_FIXED_SEED) 
        return 0;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    if (myrank == 0) {
        switch (ss->schedule_type) {
            case RING_RANDOM_SEED:
                srand((unsigned)time(NULL));
                seed = rand() % K;
                break;
            case RING_FIXED_SEED:
            default:
                break;
        }
    }
    MPI_Bcast(&seed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if(seed == ss->seed) /* no change */
        return 0;
    for (i = 0; i < K; ++i) {
        ss->sorder[i] = get_cbidx(myrank, K, i, seed); 
    }
    ss->seed = seed;
    return 1;
}
