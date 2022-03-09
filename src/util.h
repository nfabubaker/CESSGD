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

/**
 * @author      : Nabil Abubaker (nabil.abubaker@bilkent.edu.tr)
 * @file        : util
 * @created     : Thursday Aug 06, 2020 23:08:38 +03
 */

#ifndef UTIL_H

#define UTIL_H
#include "basic.h"
void substring(char *src, char *dst);
void substring_b(char *src, char *dst);

real_t dot(real_t *vec1, real_t *vec2, size_t size);
void setIDXTArrZero(idx_t *arr, idx_t size);
void setIntArrZero(int *arr, idx_t size);
void setIntArrVal(int *arr, idx_t size, int val);
void setIDXTArrVal(idx_t *arr, idx_t size, idx_t val);
void setRTArrVal(real_t *arr, idx_t size, real_t val );
void start_timer(tmr_t *t);
void stop_timer(tmr_t *t);
void shuffle_triplets(triplet *arr, idx_t size);
void shuffle(idx_t *arr, idx_t size);
void shuffle_int(int *arr, idx_t size);
void gen_perm_arr(idx_t *arr, idx_t size);
int *string_to_int_array(char *string);
#endif /* end of include guard UTIL_H */

