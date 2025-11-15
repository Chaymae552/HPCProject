#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>
#include <stdint.h>

/* Learning-rate schedule enum (for possible extensions) */
typedef enum {
    LR_CONST = 0,
    LR_TIME  = 1,
    LR_EXP   = 2,
    LR_STEP  = 3
} lr_schedule_t;

/* Activation enum shared by everything (MLP + helpers) */
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

/* ----- shuffling ----- */
void make_permutation(int *perm, int n, unsigned seed);

/* ----- dataset loader -----
   dir must contain: data_X.txt, data_y.txt
*/
int load_dataset(const char *dir,
                 double **X_out, int **y_out,
                 int *N_out, int *D_out, int *C_out);

/* ----- softmax (1 vector) ----- */
void softmax(const double *logits, int K, double *probs);

/* ----- negative log likelihood from probs ----- */
double nll_from_probs(const double *p, int y);

/* ----- activations ----- */
double act_forward(double z, act_t a);
double act_backward(double z, act_t a);

/* ----- generic LR schedule (unused in current MPI main but available) ----- */
double lr_value(lr_schedule_t sched, double lr0, double decay,
                int step_every, int t, double gamma);

#endif /* UTILS_H */
