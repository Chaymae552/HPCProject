#ifndef MODEL_H
#define MODEL_H

#include "utils.h"  /* brings act_t, lr_schedule_t, helpers */

typedef struct {
    int in_dim;
    int hdim;
    int out_dim;
    double *W1;
    double *b1;
    double *W2;
    double *b2;
} MLP;

void mlp_init(MLP *m, int in_dim, int hdim, int out_dim, unsigned seed);
void mlp_free(MLP *m);

int mlp_predict(const MLP *m, const double *x, act_t act);

void mlp_accum_grad_one(const MLP *m, const double *x, int y, act_t act,
                        double *dW1, double *db1,
                        double *dW2, double *db2,
                        double *loss_acc);

void mlp_apply_update(MLP *m, const double *dW1, const double *db1,
                      const double *dW2, const double *db2,
                      double lr, double reg, int batch_count);

void mlp_eval_range(const MLP *m, const double *X, const int *y,
                    int in_dim, int out_dim,
                    int i0, int i1,
                    act_t act,
                    long long *correct_out,
                    double *loss_sum_out);

#endif /* MODEL_H */
