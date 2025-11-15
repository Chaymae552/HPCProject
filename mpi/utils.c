#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  memory wrappers                                                   */
/* ------------------------------------------------------------------ */
void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "xmalloc(%zu) failed\n", n);
        exit(1);
    }
    return p;
}

void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "xcalloc(%zu,%zu) failed\n", n, sz);
        exit(1);
    }
    return p;
}

/* ------------------------------------------------------------------ */
/*  time in ms                                                        */
/* ------------------------------------------------------------------ */
uint64_t now_millis(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000ULL + (uint64_t)(ts.tv_nsec / 1000000ULL);
}

/* ------------------------------------------------------------------ */
/*  Fisher–Yates permutation                                          */
/* ------------------------------------------------------------------ */
void make_permutation(int *perm, int n, unsigned seed) {
    for (int i = 0; i < n; i++) perm[i] = i;

    uint64_t state = 6364136223846793005ULL * (seed + 1U) + 1ULL;

    for (int i = n - 1; i > 0; i--) {
        state = state * 6364136223846793005ULL + 1ULL;
        uint64_t r = state >> 33;
        int j = (int)(r % (uint64_t)(i + 1));

        int tmp = perm[i];
        perm[i] = perm[j];
        perm[j] = tmp;
    }
}

/* ------------------------------------------------------------------ */
/*  dataset loader: dir/{data_X.txt,data_y.txt}                       */
/* ------------------------------------------------------------------ */
int load_dataset(const char *dir,
                 double **X_out, int **y_out,
                 int *N_out, int *D_out, int *C_out)
{
    char x_path[512], y_path[512];
    snprintf(x_path, sizeof(x_path), "%s/data_X.txt", dir);
    snprintf(y_path, sizeof(y_path), "%s/data_y.txt", dir);

    int N = 0, in_dim = 2, out_dim = 2;
    char line[256];

    FILE *fx = fopen(x_path, "r");
    FILE *fy = fopen(y_path, "r");
    if (!fx || !fy) {
        fprintf(stderr, "Failed to open dataset files: '%s' and/or '%s'\n", x_path, y_path);
        if (fx) fclose(fx);
        if (fy) fclose(fy);
        return -1;
    }

    while (fgets(line, sizeof(line), fx))
        N++;
    rewind(fx);

    double *X = (double*)xmalloc((size_t)N * in_dim * sizeof(double));
    int    *y = (int*)   xmalloc((size_t)N * sizeof(int));

    double a, b;
    int t;
    for (int i = 0; i < N; i++) {
        if (fscanf(fx, "%lf %lf", &a, &b) != 2) {
            fprintf(stderr, "Bad X format at row %d\n", i);
            exit(1);
        }
        if (fscanf(fy, "%d", &t) != 1) {
            fprintf(stderr, "Bad y format at row %d\n", i);
            exit(1);
        }
        X[i*2 + 0] = a;
        X[i*2 + 1] = b;
        y[i] = t;
    }

    fclose(fx);
    fclose(fy);

    *X_out = X;
    *y_out = y;
    *N_out = N;
    *D_out = in_dim;
    *C_out = out_dim;

    return 0;
}

/* ------------------------------------------------------------------ */
/*  softmax for one vector                                            */
/* ------------------------------------------------------------------ */
void softmax(const double *logits, int K, double *probs) {
    double m = logits[0];
    for (int i = 1; i < K; i++)
        if (logits[i] > m) m = logits[i];

    double s = 0.0;
    for (int i = 0; i < K; i++) {
        probs[i] = exp(logits[i] - m);
        s += probs[i];
    }
    for (int i = 0; i < K; i++)
        probs[i] /= (s > 0.0 ? s : 1.0);
}

/* ------------------------------------------------------------------ */
/*  NLL from probs                                                    */
/* ------------------------------------------------------------------ */
double nll_from_probs(const double *p, int y) {
    double py = p[y];
    if (py < 1e-12) py = 1e-12;
    return -log(py);
}

/* ------------------------------------------------------------------ */
/*  activations                                                       */
/* ------------------------------------------------------------------ */
double act_forward(double z, act_t a) {
    switch (a) {
        case ACT_TANH:  return tanh(z);
        case ACT_RELU:  return (z > 0.0) ? z : 0.0;
        case ACT_SIGM:  return 1.0 / (1.0 + exp(-z));
        case ACT_LEAKY: return (z > 0.0) ? z : 0.01 * z;
    }
    return z;
}

double act_backward(double z, act_t a) {
    switch (a) {
        case ACT_TANH: {
            double t = tanh(z);
            return 1.0 - t*t;
        }
        case ACT_RELU:  return (z > 0.0) ? 1.0 : 0.0;
        case ACT_SIGM: {
            double s = 1.0 / (1.0 + exp(-z));
            return s * (1.0 - s);
        }
        case ACT_LEAKY: return (z > 0.0) ? 1.0 : 0.01;
    }
    return 1.0;
}

/* ------------------------------------------------------------------ */
/*  generic LR schedule (for completeness)                            */
/* ------------------------------------------------------------------ */
double lr_value(lr_schedule_t sched, double lr0, double decay,
                int step_every, int t, double gamma)
{
    switch (sched) {
        case LR_CONST: return lr0;
        case LR_TIME:  return lr0 / (1.0 + decay * (double)t);
        case LR_EXP:   return lr0 * exp(-decay * (double)t);
        case LR_STEP: {
            int k = (step_every > 0) ? (t / step_every) : 0;
            return lr0 * pow(gamma, (double)k);
        }
    }
    return lr0;
}
