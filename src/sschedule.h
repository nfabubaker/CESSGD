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
