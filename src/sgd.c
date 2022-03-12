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

#include "sgd.h"
#include "util.h"
#include "basic.h"
#include "mpi.h"
#include <stdarg.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>
#include <stdlib.h>

//#define NO_SHUFFLE

idx_t GA[GA_SIZE];

void xavier_init(real_t *arr, idx_t size, int f){
    idx_t i;
    real_t sqrf = sqrt(f);
    for (i = 0; i < size; ++i) {
        arr[i] = (real_t) rand() / (real_t) (RAND_MAX * sqrf); 
    }
}

void mat_init(real_t *arr, idx_t size, int f){
    xavier_init(arr, size, f);
}

double update_stepSize(sgd_params *params, double currEps, double prevLoss, double currLoss, int iter){
/*     if (currLoss <= prevLoss)
 *         eps *= params->eps_dec;
 *     else
 *         eps *= params->eps_inc; 
 */
    return params->init_eps/(1.0+ params->eps_dec * pow(iter, 1.5) );
}
/* NA TODO remove this function and replace it with inline ?*/
inline real_t predictRating(real_t *r_u, real_t *p_i, real_t itemBias,
        real_t q_bias, real_t r_bias, idx_t latentFactors) {
    real_t sum = 0.0;
    idx_t i;
    for (i = 0; i < latentFactors; i++) {
        sum += r_u[i] * p_i[i];
    }
    return sum + itemBias + q_bias + r_bias;
}

void update_l2(real_t *rmat, real_t *qmat, const int f, idx_t nnz, triplet *M, real_t stepSize, real_t lambda)
{
    idx_t i, nnz_idx, j;
    double err; 
    real_t *tpq, *tpr, tmp;
#ifndef NO_SHUFFLE
    gen_perm_arr(GA, nnz);
#endif /* ifndef NO_SHUFFLE */
    for (i = 0; i < nnz; i++) {
#ifndef NO_SHUFFLE
        nnz_idx = GA[i];
#else
        nnz_idx = i;
#endif /* ifndef NO_SHUFFLE */
        tpq = &qmat[ M[nnz_idx].col* f];
        tpr = &rmat[ M[nnz_idx].row * f];
        err =  M[nnz_idx].val - dot(tpr, tpq, f);
        for (j = 0; j < f; j++) { // update item vector of item i
            tmp = (*tpr);
            (*tpr) -= stepSize * ( err * (*tpq) + lambda * (*tpr));
            (*tpq) -= stepSize * ( err * tmp    + lambda * (*tpq));
            ++tpq; ++tpr;
        }
    }

}

void update_l2w(real_t *rmat, real_t *qmat, const int f, idx_t nnz, triplet *M, real_t stepSize, real_t lambda) {
    idx_t i, j, nnz_idx;
    double err, f1, f2;
    real_t *tpq, *tpr, tmp;
#ifndef NO_SHUFFLE
    gen_perm_arr(GA, nnz);
#endif /* ifndef NO_SHUFFLE */
    for (i = 0; i < nnz; i++) {
#ifndef NO_SHUFFLE
        nnz_idx = GA[i];
#else
        nnz_idx = i;
#endif /* ifndef NO_SHUFFLE */
        tpq = &qmat[ M[nnz_idx].col* f];
        tpr = &rmat[ M[nnz_idx].row * f];
        err =  M[nnz_idx].val - dot(tpr, tpq, f);
        f1 = stepSize * -2. * err;
        f2 = stepSize * 2. * lambda;
        for (j = 0; j < f; j++) { // update item vector of item i
            tmp = (*tpr);
            (*tpr) -= f1 * (*tpq) + f2 * (*tpr);
            (*tpq) -= f1 * tmp + f2 * (*tpq);
            ++tpq; ++tpr;
        }
    }
}

void _sgd_l(real_t *rmat, real_t *qmat, const int f, idx_t nnz, triplet *M, real_t stepSize, real_t lambda) {
    //update_l2(rmat, qmat, f, nnz, M, stepSize, lambda);
    if(nnz > 0)
        update_l2w(rmat, qmat, f, nnz, M, stepSize, lambda);
}

void _sgd_l_advanced(real_t *rmat, real_t *qmat, int f, idx_t nnz, triplet *M, real_t *b_r,
        real_t *b_q, real_t a, real_t b, real_t item_mean) {

    idx_t i, j,  row, col;
    real_t computedRating, err, *tpq, *tpr;
    for (i = 0; i < nnz; i++) {
        row = M[i].row;
        col = M[i].col;
        computedRating = predictRating(&rmat[row * f], &qmat[col * f], item_mean,
                b_q[col], b_r[row], f);
        err = M[i].val - computedRating;
        b_r[row] +=
            a * (err - b * b_r[row]); // update bias of user at index M[i].row
        b_q[col] +=
            a * (err - b * b_q[col]); // update bias of item at index q_lcl_idx

        tpq = &qmat[col * f];
        tpr = &rmat[row * f];
        for (j = 0; j < f; j++) { // update item vector of item i
            (*tpq) += a * (err * (*tpr) - b * (*tpq));
            (*tpr) += a * (err * (*tpq) - b * (*tpr));
            ++tpq; ++tpr;
        }
    }
}


double compute_loss_LS1(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M) {
    idx_t i,j;
    double sum = 0.0, loss, *tpq, *tpr;
    for (i = 0; i < nnz; ++i) {
        tpq = &qmat[ M[i].col* f];
        tpr = &rmat[ M[i].row * f];
        loss = (M[i].val - dot(tpr, tpq, f));
        sum += (loss * loss);
    }
    return sum;
}

double compute_loss_L2(real_t *rmat, real_t *qmat, const int f, idx_t nnz, triplet *M, double lambda) {
    idx_t i;
    int j;
    double loss = 0.0, err, newVal, qnorm, rnorm,  *tpq, *tpr;
    for (i = 0; i < nnz; ++i) {
        tpr = &rmat[M[i].row * f];
        tpq =  &qmat[M[i].col * f];
        newVal = 0.0; qnorm = 0.0; rnorm = 0.0;
        for (j = 0; j < f; ++j) {
            newVal += tpq[j] * tpr[j];
            qnorm += tpq[j] * tpq[j];
            rnorm += tpr[j] * tpr[j]; 
        }
        err = (M[i].val - newVal);
        loss += 0.5 * ((err * err) + lambda*( qnorm + rnorm ));
    }
    return loss;
}

double compute_L2w_aux(real_t *rmat, real_t *qmat, idx_t f, idx_t nrows, idx_t ncols, idx_t *nnz_per_row, idx_t *nnz_per_col, double lambda) {
    idx_t i;
    real_t  *tpq, *tpr;
    double lL2w = 0.0; 
    for (i = 0; i < nrows; ++i) {
       tpr = &rmat[i * f];
       lL2w +=  dot(tpr, tpr, f) * nnz_per_row[i];
    }
    for (i = 0; i < ncols; ++i) {
       tpq = &qmat[i * f];
       lL2w +=  dot(tpq, tpq, f) * nnz_per_col[i];
    }
    return lL2w; 
}

double compute_loss_L2w(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M, idx_t nrows, idx_t ncols, idx_t *nnz_per_row, idx_t *nnz_per_col, double lambda) {
    double lLS1, lL2w; 
    lLS1 = compute_loss_LS1(rmat, qmat, f, nnz, M);
    lL2w = compute_L2w_aux(rmat, qmat, f, nrows, ncols, nnz_per_row, nnz_per_col, lambda);
    return lLS1 + lambda*lL2w;
}

double compute_loss(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M, idx_t nrows, idx_t ncols, idx_t *nnz_per_row, idx_t *nnz_per_col, double lambda) {
    //return compute_loss_L2w(rmat, qmat, f, nnz, M,nrows, ncols, nnz_per_row, nnz_per_col, lambda);
    return 2.0*compute_loss_L2(rmat, qmat, f, nnz, M, lambda);
}

real_t compute_error(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M) {
    idx_t i,j, row, col;
    real_t sum = 0.0, computedRating, err, *tpq, *tpr;
    for (i = 0; i < nnz; ++i) {
        row = M[i].row;
        col = M[i].col;
        tpr = &rmat[row * f];
        tpq =  &qmat[col * f];
        computedRating = 0.0;
        for (j = 0; j < f; ++j) {
            computedRating += (*tpr) * (*tpq);
            tpr++; tpq++;
        }
        err = (computedRating - M[i].val);
        sum += err * err;
    }
    return sum;
}

real_t compute_error_advanced(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M,
        real_t *b_r, real_t *b_q, real_t a, real_t b,
        real_t item_mean) {
    idx_t i, row, col;
    real_t sum = 0.0, computedRating, err;
    for (i = 0; i < nnz; ++i) {
        row = M[i].row;
        col = M[i].col;
        computedRating = predictRating(&rmat[row * f], &qmat[col * f], item_mean,
                b_q[col], b_r[row], f);
        err = (M[i].val - computedRating);
        sum += err * err;
    }
    return sum;
}


