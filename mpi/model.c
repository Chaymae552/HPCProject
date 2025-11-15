#include "model.h"
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

static inline double urand_u(unsigned *st) {
    *st = *st * 1664525u + 1013904223u;
    return (*st / (double)UINT32_MAX) * 2.0 - 1.0;
}

void mlp_init(MLP *m, int in_dim, int hdim, int out_dim, unsigned seed) {
    m->in_dim  = in_dim;
    m->hdim    = hdim;
    m->out_dim = out_dim;

    m->W1 = (double*)xmalloc((size_t)in_dim * hdim    * sizeof(double));
    m->b1 = (double*)xcalloc((size_t)hdim,            sizeof(double));
    m->W2 = (double*)xmalloc((size_t)hdim   * out_dim * sizeof(double));
    m->b2 = (double*)xcalloc((size_t)out_dim,         sizeof(double));

    unsigned st = seed ? seed : 1234u;
    double s1 = sqrt(2.0 / (in_dim + hdim));
    for (int i = 0; i < in_dim * hdim; i++) m->W1[i] = s1 * urand_u(&st);

    double s2 = sqrt(2.0 / (hdim + out_dim));
    for (int i = 0; i < hdim * out_dim; i++) m->W2[i] = s2 * urand_u(&st);
}

void mlp_free(MLP *m) {
    free(m->W1); free(m->b1);
    free(m->W2); free(m->b2);
    m->W1 = m->b1 = m->W2 = m->b2 = NULL;
}

int mlp_predict(const MLP *m, const double *x, act_t act) {
    int I = m->in_dim;
    int H = m->hdim;
    int O = m->out_dim;

    double *z1 = (double*)xmalloc((size_t)H * sizeof(double));
    double *a1 = (double*)xmalloc((size_t)H * sizeof(double));
    double *z2 = (double*)xmalloc((size_t)O * sizeof(double));

    for (int j = 0; j < H; j++) {
        double s = m->b1[j];
        for (int i = 0; i < I; i++)
            s += x[i] * m->W1[i*H + j];
        z1[j] = s;
        a1[j] = act_forward(s, act);
    }

    for (int k = 0; k < O; k++) {
        double s = m->b2[k];
        for (int j = 0; j < H; j++)
            s += a1[j] * m->W2[j*O + k];
        z2[k] = s;
    }

    int arg = 0;
    double best = z2[0];
    for (int k = 1; k < O; k++)
        if (z2[k] > best) { best = z2[k]; arg = k; }

    free(z1); free(a1); free(z2);
    return arg;
}

void mlp_accum_grad_one(const MLP *m, const double *x, int y, act_t act,
                        double *dW1, double *db1, double *dW2, double *db2,
                        double *loss_acc)
{
    int I = m->in_dim;
    int H = m->hdim;
    int O = m->out_dim;

    double *z1  = (double*)xmalloc((size_t)H * sizeof(double));
    double *a1  = (double*)xmalloc((size_t)H * sizeof(double));
    double *z2  = (double*)xmalloc((size_t)O * sizeof(double));
    double *p   = (double*)xmalloc((size_t)O * sizeof(double));
    double *dz2 = (double*)xmalloc((size_t)O * sizeof(double));

    for (int j = 0; j < H; j++) {
        double s = m->b1[j];
        for (int i = 0; i < I; i++)
            s += x[i] * m->W1[i*H + j];
        z1[j] = s;
        a1[j] = act_forward(s, act);
    }

    for (int k = 0; k < O; k++) {
        double s = m->b2[k];
        for (int j = 0; j < H; j++)
            s += a1[j] * m->W2[j*O + k];
        z2[k] = s;
    }

    softmax(z2, O, p);
    *loss_acc += nll_from_probs(p, y);

    for (int k = 0; k < O; k++)
        dz2[k] = p[k] - (k == y ? 1.0 : 0.0);

    for (int k = 0; k < O; k++) {
        db2[k] += dz2[k];
        for (int j = 0; j < H; j++)
            dW2[j*O + k] += a1[j] * dz2[k];
    }

    for (int j = 0; j < H; j++) {
        double s = 0.0;
        for (int k = 0; k < O; k++)
            s += m->W2[j*O + k] * dz2[k];
        double dz1 = s * act_backward(z1[j], act);
        db1[j] += dz1;
        for (int i = 0; i < I; i++)
            dW1[i*H + j] += x[i] * dz1;
    }

    free(z1); free(a1); free(z2); free(p); free(dz2);
}

void mlp_apply_update(MLP *m, const double *dW1, const double *db1,
                      const double *dW2, const double *db2,
                      double lr, double reg, int batch_count)
{
    int I = m->in_dim;
    int H = m->hdim;
    int O = m->out_dim;
    double inv = 1.0 / (double)batch_count;

    for (int j = 0; j < H; j++) {
        m->b1[j] -= lr * (db1[j] * inv);
        for (int i = 0; i < I; i++) {
            double g = dW1[i*H + j]*inv + reg * m->W1[i*H + j];
            m->W1[i*H + j] -= lr * g;
        }
    }
    for (int k = 0; k < O; k++) {
        m->b2[k] -= lr * (db2[k] * inv);
        for (int j = 0; j < H; j++) {
            double g = dW2[j*O + k]*inv + reg * m->W2[j*O + k];
            m->W2[j*O + k] -= lr * g;
        }
    }
}

void mlp_eval_range(const MLP *m, const double *X, const int *y,
                    int in_dim, int out_dim,
                    int i0, int i1,
                    act_t act,
                    long long *correct_out,
                    double *loss_sum_out)
{
    long long correct = 0;
    double loss_sum = 0.0;

    for (int i = i0; i < i1; i++) {
        const double *x = &X[i*in_dim];
        int pred = mlp_predict(m, x, act);
        if (pred == y[i]) correct++;

        double *z1 = (double*)xmalloc((size_t)m->hdim * sizeof(double));
        double *a1 = (double*)xmalloc((size_t)m->hdim * sizeof(double));
        double *z2 = (double*)xmalloc((size_t)m->out_dim * sizeof(double));
        double *p  = (double*)xmalloc((size_t)m->out_dim * sizeof(double));

        for (int j = 0; j < m->hdim; j++) {
            double s = m->b1[j];
            for (int ii = 0; ii < m->in_dim; ii++)
                s += x[ii] * m->W1[ii*m->hdim + j];
            z1[j] = s;
            a1[j] = act_forward(s, act);
        }
        for (int k = 0; k < m->out_dim; k++) {
            double s = m->b2[k];
            for (int j = 0; j < m->hdim; j++)
                s += a1[j] * m->W2[j*m->out_dim + k];
            z2[k] = s;
        }
        softmax(z2, m->out_dim, p);
        loss_sum += nll_from_probs(p, y[i]);

        free(z1); free(a1); free(z2); free(p);
    }

    *correct_out = correct;
    *loss_sum_out = loss_sum;
}
