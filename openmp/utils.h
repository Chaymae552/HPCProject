// utils.h  — header used by OpenMP build
#ifndef UTILS_H
#define UTILS_H

#include <stddef.h>

// ---------- Learning-rate schedules ----------
typedef enum {
    LR_CONST = 0,   // η(t) = η0
    LR_TIME  = 1,   // η(t) = η0 / (1 + decay*t)
    LR_EXP   = 2,   // η(t) = η0 * exp(-decay*t)
    LR_STEP  = 3    // η(t) = η0 * gamma^( floor(t/step_every) )
} lr_schedule_t;

// Public LR API (epoch/step-based policy)
double lr_value_iter(lr_schedule_t sched, double lr0, double decay,
                int epoch_or_step, int step_every, double gamma);

// ---------- Safe alloc ----------
void* xmalloc(size_t nbytes);
void* xcalloc(size_t n, size_t sz);

// ---------- Random ----------
double randn(void);

// ---------- BLAS-free kernels (row-major) ----------
void matmul(double *A, double *B, double *C, int n, int m, int p);
void add_bias(double *Z, double *b, int n, int p);

// Row-wise softmax (matrix): inputs Z[n x p] -> P[n x p]
void softmax(double *Z, double *P, int n, int p);

// ---------- Dataset I/O ----------
int  count_lines(const char *filename);
void load_X(const char *filename, double *X, int num_examples, int input_dim);
void load_y(const char *filename, int *y, int num_examples);

#endif // UTILS_H
