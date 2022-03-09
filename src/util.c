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


#include "util.h"
#include "basic.h"
#include <stdlib.h>
#include <string.h>
#define SWAP(T, a, b) do { T tmp = a; a = b; b = tmp; } while (0)
void substring(char *dst, char *src){
    char *ptr = src;
    char *prevptr = NULL;
    while( (ptr = strstr(ptr, "/"))){
        prevptr = ptr++;
    }

    prevptr++;
    int sl = strlen(prevptr);
    strncpy(dst, prevptr, sl);
    dst[sl] = '\0';
}

void shuffle_int(int *arr, idx_t size){
    long int i,j;
    int tmp;
    for (i = (long int)size-1; i >= 0; --i) {
        srand((unsigned)time(NULL));
        j = rand() % (i+1);
        tmp = arr[i];
        arr[i] = arr[j]; arr[j] = tmp;
    }
}

int *string_to_int_array(char *string){
    int i, size = 1;
    int *arr;
    for(i = 0; string[i] != '\0'; i++){
        if(string[i] == ','){ 
            size++;
           // printf("Hey I found a comma!!\n");
        }
    }

    arr = malloc(sizeof(int) * (size+1));
    for (i = 0; i <= size; ++i) {
        arr[i] = 0;
    }
    arr[0] = size;
    int j = 1;
    for(i = 0; string[i] != '\0'; i++){
        if(string[i] == ',') j++;
        else
            arr[j] = 10*arr[j] + (string[i] - 48);
    }
    return arr;
}


void shuffle(idx_t *arr, idx_t size){
    long int i,j;
    idx_t tmp;
    for (i = size-1; i >= 0; --i) {
        srand((unsigned)time(NULL));
        j = rand() % (i+1);
        tmp = arr[i];
        arr[i] = arr[j]; arr[j] = tmp;
    }
}

void gen_perm_arr(idx_t *arr, idx_t size){
    idx_t i;
    for (i = 0; i < size; ++i) {
        arr[i] = i;
    }
    shuffle(arr, size);
}

void shuffle_triplets(triplet *arr, idx_t sz){
    long int i,j;
    for (i = sz-1; i >= 0; i--) {
        srand((unsigned)time(NULL));
        j = rand() % (i+1);
        SWAP(triplet, arr[i], arr[j]);
    }
}

real_t dot(real_t *vec1, real_t *vec2, size_t size){
    real_t sum = 0.0; 
    size_t i;
    for (i = 0; i < size; ++i) {
        sum += vec1[i] * vec2[i];
    }
    return sum;
}

void substring_b(char *dst, char *src){
    char *ptr = src;
    char *prevptr = NULL;
    
    while( (ptr = strstr(ptr, "."))){
        prevptr = ptr++;
    }

    //prevptr;
    int sl = strlen(src) - strlen(prevptr);
    strncpy(dst, src, sl);
    dst[sl] = '\0';
}
void setIDXTArrZero(idx_t *arr, idx_t size){
    idx_t i;
    for (i = 0; i < size; ++i) {
        arr[i] = 0;
    }
}
void setIntArrZero(int *arr, idx_t size){
    idx_t i;
    for (i = 0; i < size; ++i) {
        arr[i] = 0;
    }
}
void setIDXTArrVal(idx_t *arr, idx_t size, idx_t val){
    idx_t i;
    for (i = 0; i < size; ++i) {
        arr[i] = val;
    }
}
void setIntArrVal(int *arr, idx_t size, int val){
    idx_t i;
    for (i = 0; i < size; ++i) {
        arr[i] = val;
    }
}
void setRTArrVal(real_t *arr, idx_t size, real_t val ){
    size_t i;
    for (i = 0; i < size; ++i) {
        arr[i] = val;  
    }
}



void start_timer(tmr_t *t){
    clock_gettime(CLOCK_MONOTONIC, &t->ts_beg);
    return;
}

void stop_timer(tmr_t *t){
    clock_gettime(CLOCK_MONOTONIC, &t->ts_end);
    t->elapsed += 1000000000.0 * (double) (t->ts_end.tv_sec - t->ts_beg.tv_sec) + (double) (t->ts_end.tv_nsec - t->ts_beg.tv_nsec);
    return;
}
