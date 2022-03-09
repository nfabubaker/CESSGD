/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : sschedule
 * @created     : Saturday Nov 20, 2021 14:41:32 +03
 */

#ifndef SSCHEDULE_H

#define SSCHEDULE_H
#include "basic.h"
/* STRATA schedule  */
/*! \enum SSCHEDULE_TYPE
 *
 *  Detailed description
 */
enum SSCHEDULE_TYPE { RING_FIXED_SEED, RING_RANDOM_SEED};
typedef struct _sschedule{
    int *order;
    int *sorder;//which Column stripe I will be updating at SE i 
    int *odist;
    int seed;
    int schedule_type;
} sschedule;

void free_ss(sschedule *ss);
int get_se(int pid, int K, int CS, int seed);
int get_cbidx(int pid, int K, int SE, int seed);
void init_sschedule(sschedule *ss, int K);
int update_sschedule(sschedule *ss, int K);
#endif /* end of include guard SSCHEDULE_H */
