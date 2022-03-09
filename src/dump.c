
void SM_S_afterUpdate(Comm *cm, int *col_update_order, genst *gs){
    {
        int i, j, tmpcnt, pid, nsendto, *xsendinds, *sendList, *sendinds, *tcnts, *gtlmap;
        tcnts = malloc(gs->nprocs * sizeof(*tcnts));
        gtlmap = malloc(gs->nprocs * sizeof(*gtlmap));
        cm->nsend = malloc(gs->nprocs * sizeof(*cm->nsend));
        /* for each stratum */
        for (i = 0; i < gs->nprocs; ++i) {
            setIntArrVal(gtlmap, gs->nprocs, -1);
            setIntArrZero(tcnts, gs->nprocs); 
            nsendto = 0;
#ifdef NA_DBG

            na_log(dbgfp, "\t\tin Schedule messages, just before counting, nprocs=%d\n", gs->nprocs);
#endif
            /* loop over each stratum's local columns and determine which processors update them first*/
            for (j = gs->xlcols[i]; j < gs->xlcols[i+1]; ++j) {
                pid = col_update_order[gs->lcols[j]];
#ifdef NA_DBG
                assert(pid < gs->nprocs);
#endif
                if (pid != -1) { /* if someone updates the column */
                    if(tcnts[pid] == 0)
                        nsendto++;
                    tcnts[pid]++; 
                }
            }
#ifdef NA_DBG

            na_log(dbgfp, "\t\tin Schedule messages, done counting first update for stratum %d, nsendto=%d\n", i, nsendto);
#endif
            cm->nsend[i] = nsendto;
            if (nsendto > 0) {
                cm->sendList[i]= malloc(sizeof(*cm->sendList[i]) * nsendto);
                sendList = cm->sendList[i];

#ifdef NA_DBG

                na_log(dbgfp, "\t\tdbgpt 1.0\n"); 
#endif
                //cm->xsendinds[i] = calloc((nsendto+2), sizeof(*cm->xsendinds[i]));
                cm->xsendinds[i] = malloc(sizeof(*cm->xsendinds[i]) * (nsendto+2));
                xsendinds = cm->xsendinds[i];
                setIntArrZero(xsendinds, nsendto+2);
#ifdef NA_DBG

                na_log(dbgfp, "\t\tdbgpt 1.1\n"); 
#endif
                /* Fill the sendList and xsendinds arrays*/
                tmpcnt = 0; 
                for (j = 0; j < gs->nprocs; ++j) {
                    if(tcnts[j] > 0){
                        sendList[tmpcnt] = j;
                        xsendinds[tmpcnt+2] = tcnts[j];
                        gtlmap[j] = tmpcnt;
                        tmpcnt++;
                    }
                }
#ifdef NA_DBG
                na_log(dbgfp, "\t\tnsendto=%d, tmpcn=%d\n", nsendto, tmpcnt); 
                assert(tmpcnt == nsendto);

                na_log(dbgfp, "\t\tdbgpt 1.2\n"); 
#endif
                for (j = 2; j < nsendto+2; ++j) {
                    xsendinds[j] += xsendinds[j-1];
                }
#ifdef NA_DBG
                na_log(dbgfp, "\t\txsendinds created and filled for stratum %d, nsendto=%d, nmsgs=%d\n", i, nsendto, xsendinds[nsendto+1]);
#endif
                cm->sendinds[i] = malloc(sizeof(*cm->sendinds[i]) * xsendinds[nsendto+1]);
                sendinds = cm->sendinds[i];
#ifdef NA_DBG
                na_log(dbgfp, "\t\tdbgpt 1.3\n"); 
#endif
                /* fill the local indices to be sent to each processor */
                for (j = gs->xlcols[i]; j < gs->xlcols[i+1]; ++j) {
                    pid = col_update_order[gs->lcols[j]];
                    if (pid != -1) {
                        assert(gtlmap[pid] != -1);
                        sendinds[ xsendinds[ gtlmap[pid] + 1 ]++ ] = j; 
                    }
                }
#ifdef NA_DBG
                na_log(dbgfp, "\t\tsendinds created and filled for stratum %d\n", i);
#endif
                assert(xsendinds[nsendto]==xsendinds[nsendto+1]);
                /* reset temp arrays */
            }
        }
        /* cleanup */
        free(tcnts); free(gtlmap);

    }
