#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

/* Learning-rate schedule enum */
typedef enum {
    LR_CONST = 0,
    LR_TIME  = 1,
    LR_EXP   = 2,
    LR_STEP  = 3
} lr_schedule_t;

/* Activation enum shared by everything */
typedef enum {
    ACT_TANH  = 0,
    ACT_RELU  = 1,
    ACT_SIGM  = 2,
    ACT_LEAKY = 3
} act_t;

/* ----- memory wrappers ----- */
void *xmalloc(size_t n);
void *xcalloc(size_t n, size_t sz);

/* ----- timing ----- */
uint64_t now_millis(void);

void make_permutation(int *perm, int n, unsigned seed);

int load_dataset(const char *dir,
                 double **X_out, int **y_out,
                 int *N_out, int *D_out, int *C_out);

void softmax(const double *logits, int K, double *probs);

double nll_from_probs(const double *p, int y);

/* ----- activations ----- */
double act_forward(double z, act_t a);
double act_backward(double z, act_t a);

/* ----- LR schedule ----- */
double lr_value(lr_schedule_t sched, double lr0, double decay,
                int step_every, int t, double gamma);

#endif
