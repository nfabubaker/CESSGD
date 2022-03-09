#ifndef SGD_H_19DIUSCZ
#define SGD_H_19DIUSCZ

#include "basic.h"

typedef struct _sgd_params{
    double init_eps, eps, lambda, eps_inc, eps_dec;
} sgd_params;

void xavier_init(real_t *arr, idx_t size, int f);
void mat_init(real_t *arr, idx_t size, int f);
real_t compute_error_advanced(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M,
                     real_t *b_r, real_t *b_q, real_t a, real_t b,
                     real_t item_mean);

real_t compute_error(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M);
/* NA TODO remove this function and replace it with inline ?*/
real_t predictRating(real_t *r_u, real_t *p_i, real_t itemBias,
                            real_t q_bias, real_t r_bias, idx_t latentFactors);

double update_stepSize(sgd_params *params, double currEps, double prevLoss, double currLoss, int iter);
double compute_loss(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M, idx_t nrows, idx_t ncols,idx_t *nnz_per_row, idx_t *nnz_per_col, double lambda);
double compute_loss_L2(real_t *rmat, real_t *qmat, const int f, idx_t nnz, triplet *M, double lambda);
double compute_L2w_aux(real_t *rmat, real_t *qmat, idx_t f, idx_t nrows, idx_t ncols, idx_t *nnz_per_row, idx_t *nnz_per_col, double lambda);
double compute_loss_LS1(real_t *rmat, real_t *qmat, idx_t f, idx_t nnz, triplet *M);

void _sgd_l(real_t *rmat, real_t *qmat, int f, idx_t nnz, triplet *M, real_t stepSize, real_t lambda);
void _sgd_l_advanced(real_t * rmat, real_t * qmat, int f, idx_t nnz, triplet *M,
                real_t *b_r, real_t *b_q, real_t a, real_t b,
                real_t item_mean );


#endif /* end of include guard: SGD_H_19DIUSCZ */
