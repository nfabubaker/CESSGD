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

