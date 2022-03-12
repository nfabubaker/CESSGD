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

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <getopt.h>
#include <stdarg.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include "def.h"
#include "comm.h"
#include "io.h"
#include "sgd.h"
#include "basic.h"
#include "util.h"
#include "sschedule.h"
#include <assert.h>

#define WA_ITER 2

#ifdef TAKE_TIMES

tmr_t setuptime    = {0};
tmr_t sgdtime      = {0};
tmr_t preptime      = {0};
tmr_t commtime     = {0};
tmr_t lsgdtime     = {0};

void print_times(idx_t niter) {

    double maxsetuptime, maxsgdtime, maxcommtime, maxtot, maxlsgdtime, maxpreptime; 
    double total = setuptime.elapsed + sgdtime.elapsed; 
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    MPI_Reduce(&lsgdtime.elapsed, &maxlsgdtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&commtime.elapsed, &maxcommtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&preptime.elapsed, &maxpreptime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&sgdtime.elapsed, &maxsgdtime, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&setuptime.elapsed, &maxsetuptime, 1, MPI_DOUBLE, MPI_MAX, 0,
            MPI_COMM_WORLD);
    MPI_Reduce(&total, &maxtot, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    double n = (double)niter;
    if (myrank == 0) {
        double mult = 1e-6; //for ms 
        printf("%s\n", "SetupTime CommPrepTime compTime CommTime sgdIterationTime totTime:");
        printf("%.5f %.5f %.5f %.5f %.5f %.5f\n\n", maxsetuptime*mult, maxpreptime*mult/n, maxlsgdtime*mult /n , maxcommtime*mult/n, maxsgdtime*mult / n, maxtot*mult);
    }

}
#endif
#ifdef PRINT_STAT
void print_stat(const ldata *gs, const sschedule *ss, const Comm *cm){
    int i, nprocs, myrank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    /* First, total msgs */

    unsigned int totMsgs = 0, maxmaxMsgs =0, summaxMsgs=0, gmaxmaxMsgs, gsummaxMsgs, gtotMsgs, totVol=0, lmaxVM[2], gsummaxVol, gmaxmaxVol, gtotVol, sumOfMaxNnz, maxNnzPerSE;

    sumOfMaxNnz = 0;
    for (i = 0; i < nprocs; ++i) {
        MPI_Reduce(&(gs->nnz_per_stratum[ss->sorder[i]]), &maxNnzPerSE, 1, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
        sumOfMaxNnz+= maxNnzPerSE;
    }
    if(cm->comm_type==0){
        idx_t maxColStrip;
        gtotMsgs = nprocs*nprocs;
        gtotVol = gs->ngcols * nprocs;
        gmaxmaxMsgs = 1; gsummaxMsgs = nprocs; 
        gsummaxVol = gs->maxColStrip * nprocs;
        gmaxmaxVol = gs->maxColStrip;
    }
    else{
        idx_t ttmp[2];
        //MPI_Reduce(cm->nsend, gnsends, nprocs, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
        gsummaxMsgs = 0;
        gsummaxVol = 0;
        gmaxmaxMsgs = 0;
        gmaxmaxVol = 0;
        for (i = 0; i < nprocs; ++i) { //for each SE
            totMsgs += cm->nsend[i];
            if(cm->nsend[i] > maxmaxMsgs)
                maxmaxMsgs = cm->nsend[i];
            totVol += cm->xsendinds[i][cm->nsend[i]];
            ttmp[0] = cm->xsendinds[i][cm->nsend[i]];
            ttmp[1] = cm->nsend[i];
            MPI_Reduce(ttmp, lmaxVM, 2, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
            gsummaxMsgs += lmaxVM[1];
            gsummaxVol += lmaxVM[0];
            if(lmaxVM[0] > gmaxmaxVol)
                gmaxmaxVol = lmaxVM[0];
            if(lmaxVM[1] > gmaxmaxMsgs)
                gmaxmaxMsgs = lmaxVM[1];
        }
        MPI_Reduce(&totMsgs, &gtotMsgs, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&totVol, &gtotVol, 1, MPI_UNSIGNED, MPI_SUM, 0, MPI_COMM_WORLD);
        //        MPI_Reduce(&maxmaxMsgs, &gmaxmaxMsgs, 1, MPI_UNSIGNED, MPI_MAX, 0, MPI_COMM_WORLD);
    }
    if(myrank == 0)
        printf("%s\n", "mmMsg smMsg totMsg mmVol smVol totVol smNnzRatio, localCols/globalCols");
        printf("%u %u %u %u %u %u %.5f %.5f\n\n", gmaxmaxMsgs, gsummaxMsgs, gtotMsgs, gmaxmaxVol, gsummaxVol, gtotVol, sumOfMaxNnz/(gs->gnnz/(double)nprocs), gs->nlcols/(double)gs->ngcols);

} 
#endif

void print_usage(char *exec) {
    printf("Usage: %s [options] <Matrix file>\n "
            "Options:\n",
            exec);

    printf("\t-i max. number of iterations (epochs), default: 10\n");
    printf("\t-f number of latent factors, default: 16 NOTE: give as a srtring e.g \"30\"\n");
    printf("\t-l regularization factor (lambda, default=0.0075)\n");
    printf("\t-e learning rate (eps, default=0.0015)\n");
    printf("\t-c Communication type:\n"
            "\t\t0: Block-wise (DSGD)\n"
            "\t\t1: P2P\n"
            "\t\t3: P2P with Hold and Combine\n");
    printf("\t-s Strata schedule type:"
            "\t\t0: RING_FIXED_SEED (default) picks a seed randomly at first epoch and sticks to it in all epochs\n"
            "\t\t1: RING_RANDOM_SEED picks a different seed randomly at each epoch\n");
    printf("\t-p Partition file, if not provided random partitioning of rows and cols is used. \n");
    printf("\t-d Force random partition of column even if a partition file is provided\n");
    printf("\t-h Print this help message\n");
}
void init_genst(genst *gs){
    gs->f = 16;
    gs->niter = 10;
    gs->nnz = 0;
    gs->gnnz = 0;
    gs->comm_type = AFTER_UPDATE;
    gs->use_pfile = 0;
    gs->use_randColDist = 1;
    gs->lambda = 0.0075;
    gs-> eps = 0.0015;
    gs->sschedule_type = RING_FIXED_SEED;
}
void init_params(int argc, char *argv[], genst *gs){
    int choice;
   char tt[512];
    while (1)
    {
        static struct option long_options[] =
        {
            /* Use flags like so:
               {"verbose",	no_argument,	&verbose_flag, 'V'}*/
            /* Argument styles: no_argument, required_argument, optional_argument */
            {"version", no_argument,	0,	'v'},
            {"help",	no_argument,	0,	'h'},

            {0,0,0,0}
        };

        int option_index = 0;

        /* Argument parameters:
no_argument: " "
required_argument: ":"
optional_argument: "::" */

        choice = getopt_long( argc, argv, "vh:i:f:c:p:t:e:s:l:d",
                long_options, &option_index);

        if (choice == -1)
            break;

        switch( choice )
        {
            case 'v':

                break;
            case 'i':
                gs->niter = atoi(optarg);
                break;
            case 'f':
                sprintf(tt, "%s", optarg);
                gs->fVals = string_to_int_array(tt);
                gs->f = gs->fVals[1];
                break;
            case 'c':
                gs->comm_type = atoi(optarg);
                break;
            case 'h':
                print_usage(argv[0]);
                break;
            case 'l':
                gs->lambda = atof(optarg);
                break;
            case 'e':
                gs->eps = atof(optarg);
                break;
            case 's':
                gs->sschedule_type = atoi(optarg);
                switch (gs->sschedule_type) {
                    case 0:
                        gs->sschedule_type = RING_FIXED_SEED;
                        break;
                    case 1:
                        gs->sschedule_type = RING_RANDOM_SEED;
                        break;
                    default:
                        fprintf(stderr,"ERROR: incorrect strata schedule choice\n");
                        exit(EXIT_FAILURE);
                }
                break;
            case 'p':
                gs->use_pfile = 1;
                gs->use_randColDist = 0;
                sprintf(gs->pvecFN, "%s", optarg);
                break;
            case 'd':
                gs->use_randColDist = 1;
                break;
            case '?':
                /* getopt_long will have already printed an error */
                break;

            default:
                /* Not sure how to get here... */
                break;
        }
    }

    /* Deal with non-option arguments here */

    if (optind >= argc || argc - optind < 1) {
        fprintf(stderr, "Wrong Execution !!\n");
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }
    if (optind < argc) {
        sprintf(gs->mtxFN, "%s", argv[optind]);
    }
}

/* 
 * TODO This is inefficient implementation, the norms of Q and R matrix rows are computed with 
 * nonzeros while they can be computed seperately, see NOMAD paper page 2.
 * */
double compute_loss_L2w_par(ldata *gs, double lambda) {
    idx_t i;
    double   lL2w, gL2w;
    lL2w = 0.0;
    for (i = 0; i < gs->nprocs; ++i) {
        lL2w += compute_loss_L2(gs->rmat, gs->qmat, gs->f, gs->nnz_per_stratum[i], gs->mtx[i], lambda);
    }
    MPI_Allreduce(&lL2w, &gL2w, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return gL2w;
}
double compute_loss_par(ldata *gs, double lambda) {
    return compute_loss_L2w_par(gs,  lambda);
}

real_t computeDSGD(ldata *lData, sgd_params *params, Comm *cm, sschedule *ss, double initLoss, int nMaxIter, int startingIter)
{
    int notconverged = 1, niter = startingIter, cbidx, sid;
    double  prevLoss = 0.0, currLoss = 0.0;

    cm->epochID = 0;
    currLoss = initLoss;
    /* main loop */
    while (notconverged && niter < nMaxIter) {
#ifdef TAKE_TIMES
        start_timer(&preptime);
#endif
        if (niter != 0){
            params->eps = update_stepSize(params, params->eps, prevLoss, currLoss, niter);
            prepare_comm(cm, lData, ss);
        }
        else
            params->eps = params->init_eps;

#ifdef TAKE_TIMES
        stop_timer(&preptime);
#endif
        for (sid = 0; sid < lData->nprocs; ++sid) { // for each stratum
            cbidx = ss->sorder[sid]; 
#ifdef TAKE_TIMES
            start_timer(&lsgdtime);
#endif
            _sgd_l(lData->rmat, lData->qmat,  lData->f, lData->nnz_per_stratum[cbidx], lData->mtx[cbidx], params->eps, params->lambda);
#ifdef TAKE_TIMES
            stop_timer(&lsgdtime);
            start_timer(&commtime);
#endif
            communicate_qmat_rows(lData->qmat, lData, cm, ss, sid);
#ifdef TAKE_TIMES
            stop_timer(&commtime);
#endif
        }
        prevLoss = currLoss;
        currLoss = compute_loss_par(lData, params->lambda);
#ifdef TAKE_TIMES
        stop_timer(&sgdtime);
        double maxsgdtime;
        MPI_Reduce(&sgdtime.elapsed, &maxsgdtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        if (cm->myrank == 0) {
            printf("[i%u] %f %f %f\n",niter, params->eps, currLoss, maxsgdtime * 1e-6);
        }
        start_timer(&sgdtime);
#endif
        cm->epochID++;
        niter++;
    }
    return currLoss; 
}


void na_log(FILE *fp, const char *format, ...) {
    va_list args;
    va_start(args, format);
    vfprintf(fp, format, args);
    fflush(fp);
    va_end(args);
}


void gen_rand_pvec(int * const rpvec, int * const colpvec, const int ngrows, const int ngcols, const genst *gs){
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

    if (gs->myrank == 0) {
            idx_t i, tcnt = 0;
            const int nlarger = ngrows > ngcols ? ngrows : ngcols;
            idx_t *permM = malloc(nlarger * sizeof(*permM));
            gen_perm_arr(permM, ngrows);
            if(gs->use_pfile == 0){
                for (i = 0; i < ngrows; ++i) {
                    rpvec[permM[i]] = (int)(tcnt++  % gs->nprocs);
                }
            }
            gen_perm_arr(permM, ngcols);
            if(gs->use_randColDist == 1){
                tcnt = 0;
                for (i = 0; i < ngcols; ++i) {
                    colpvec[permM[i]] = (tcnt++  % gs->nprocs);
                }
            }
    }
    if(gs->use_pfile == 0){
        MPI_Bcast(rpvec, ngrows, MPI_INT, 0, MPI_COMM_WORLD);
    }
    if(gs->use_randColDist == 1){
        MPI_Bcast(colpvec, ngcols, MPI_INT, 0, MPI_COMM_WORLD);
    }
}

void start_sgd_instance(triplet *M, ldata *lData, Comm *cm, sschedule *ss, genst *gs, const int *rpvec, const int *colpvec, char *name){
    double initLoss, tmpLoss, finalLoss;
#ifdef TAKE_TIMES
    setuptime.elapsed    = 0;
    sgdtime.elapsed      = 0;
    commtime.elapsed             = 0;
    preptime.elapsed             = 0;
    lsgdtime.elapsed             = 0;
    start_timer(&(setuptime));
#endif
    /* initialize per stratum matrix, local indxs .. etc */
    ss->schedule_type = gs->sschedule_type;
    init_sschedule(ss, gs->nprocs);
    setup_ldata(M, lData, colpvec, gs->comm_type, gs->f);
    setup_comm(cm, lData, ss, colpvec, gs->comm_type, gs->use_randColDist);

#ifdef TAKE_TIMES
    stop_timer(&(setuptime));
#endif
    sgd_params params;
    params.init_eps = gs->eps; params.lambda = gs->lambda; 
    params.eps_inc = 1.05; params.eps_dec = 0.05;
    /* Warmup */
    initLoss = compute_loss_par(lData, params.lambda);
    /*     tmpLoss = computeDSGD(&lData, &params, &cm, &ss, initLoss , WA_ITER, 0);
    */
#ifdef TAKE_TIMES
    commtime.elapsed             = 0;
    preptime.elapsed             = 0;
    lsgdtime.elapsed             = 0;
    MPI_Barrier(MPI_COMM_WORLD);
    start_timer(&sgdtime);
#endif
    finalLoss = computeDSGD(lData, &params, cm, ss, initLoss, gs->niter, 0);
#ifdef TAKE_TIMES
    stop_timer(&sgdtime);
#endif
    if (gs->myrank == 0) {
        printf("%s\n","MatrixName MPI-ranks CommType f initLoss finalLoss LossDiff"); 
        printf("%s %d %d %d %f %f %f\n\n", name, gs->nprocs, gs->comm_type, gs->f, initLoss, finalLoss, initLoss - finalLoss); 
    }
#ifdef TAKE_TIMES
    print_times(gs->niter); 
#endif

#ifdef PRINT_STAT
    print_stat(lData, ss, cm);
#endif
    if(gs->myrank == 0)
        printf("\n");
    free_comm(cm);
    free_lData(lData);
    free_ss(ss);

}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    genst gs;
    triplet *M;
    Comm cm;
    ldata lData;
    sschedule ss;
    int i, *rpvec, *colpvec;
    init_genst(&gs);
    MPI_Comm_rank(MPI_COMM_WORLD, &gs.myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &gs.nprocs);
    init_params(argc, argv, &gs);
    char fname[1024], name[1024];
    substring(fname, gs.mtxFN);
    substring_b(name, fname);
    /* read and distr input matrix  */
    read_metadata(gs.mtxFN, &lData.ngrows, &lData.ngcols, &lData.gnnz);
    rpvec = malloc(lData.ngrows * sizeof(*rpvec));
    colpvec = malloc(lData.ngcols * sizeof(*colpvec));
    if (gs.use_pfile) {
        read_partvec_bc(gs.pvecFN, rpvec, colpvec, lData.ngrows, lData.ngcols, gs.use_randColDist);
    }
    gen_rand_pvec(rpvec, colpvec, lData.ngrows, lData.ngcols, &gs);
    init_lData(&lData);
    read_matrix_bc(gs.mtxFN, &M, rpvec, &lData);
    int comm_type = gs.comm_type;
    for (i = 0; i < gs.fVals[0]; ++i) {
        gs.f = gs.fVals[i+1];
        switch (comm_type) {
            case 0:
            case 1:
            case 2:
            case 3:
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                break;
            case 4: /* 0 & 1  */
                gs.comm_type = NAIIVE;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                gs.comm_type = AFTER_UPDATE;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                break;
            case 5: /* 1 & 3 */
                gs.comm_type = AFTER_UPDATE;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                gs.comm_type = SMART;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                break;
            case 6: /* 0,1 and 3 */
                gs.comm_type = NAIIVE;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                gs.comm_type = AFTER_UPDATE;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                gs.comm_type = SMART;
                start_sgd_instance(M, &lData, &cm, &ss, &gs, rpvec, colpvec, name);
                break;
            default: 
                fprintf(stderr, "Comm type choice error!\n");
                exit(EXIT_FAILURE);
                break;
        }
    }
    free(M);
    MPI_Finalize();
    return 0;
}
