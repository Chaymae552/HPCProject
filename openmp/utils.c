// utils.c — plain C implementations + OpenMP (no BLAS)
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

/* ---------- Safe Allocation ---------- */
void* xmalloc(size_t nbytes) {
    void *p = malloc(nbytes);
    if (!p) {
        fprintf(stderr, "malloc failed (%lu bytes)\n", (unsigned long)nbytes);
        exit(EXIT_FAILURE);
    }
    return p;
}
void* xcalloc(size_t n, size_t sz) {
    void *p = calloc(n, sz);
    if (!p) {
        fprintf(stderr, "calloc failed (%lu x %lu)\n",
                (unsigned long)n, (unsigned long)sz);
        exit(EXIT_FAILURE);
    }
    return p;
}

/* ---------- Random (Box–Muller) ---------- */
double randn(void) {
    static int hasSpare = 0;
    static double spare;
    if (hasSpare) { hasSpare = 0; return spare; }

    double u, v, s;
    do {
        u = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        v = (rand() / ((double)RAND_MAX)) * 2.0 - 1.0;
        s = u*u + v*v;
    } while (s >= 1.0 || s == 0.0);

    s = sqrt(-2.0 * log(s) / s);
    spare = v * s;
    hasSpare = 1;
    return u * s;
}

/* ---------- Matrix Operations (OpenMP) ---------- */
// Row-major: C[n x p] = A[n x m] * B[m x p]
void matmul(double *A, double *B, double *C, int n, int m, int p) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            double sum = 0.0;
            const int arow = i * m;
            for (int k = 0; k < m; k++)
                sum += A[arow + k] * B[k*p + j];
            C[i*p + j] = sum;
        }
    }
}

void add_bias(double *Z, double *b, int n, int p) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < p; j++)
            Z[i*p + j] += b[j];
}

// Row-wise softmax with max-shift for numerical stability
void softmax(double *Z, double *P, int n, int p) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double maxz = Z[i*p + 0];
        for (int j = 1; j < p; j++) {
            double v = Z[i*p + j];
            if (v > maxz) maxz = v;
        }
        double sum = 0.0;
        for (int j = 0; j < p; j++) {
            double e = exp(Z[i*p + j] - maxz);
            P[i*p + j] = e;
            sum += e;
        }
        const double inv = 1.0 / sum;
        for (int j = 0; j < p; j++) P[i*p + j] *= inv;
    }
}

/* ---------- Dataset I/O ---------- */
int count_lines(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erreur ouverture fichier %s: %s\n", filename, strerror(errno));
        return -1;
    }
    int lines = 0, c;
    while ((c = fgetc(fp)) != EOF)
        if (c == '\n') ++lines;
    fclose(fp);
    return lines;
}

void load_X(const char *filename, double *X, int num_examples, int input_dim) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erreur ouverture X (%s): %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_examples; i++) {
        for (int j = 0; j < input_dim; j++) {
            if (fscanf(fp, "%lf", &X[i*input_dim + j]) != 1) {
                fprintf(stderr, "Erreur lecture X à la ligne %d colonne %d dans %s\n",
                        i+1, j+1, filename);
                fclose(fp);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(fp);
}

void load_y(const char *filename, int *y, int num_examples) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Erreur ouverture y (%s): %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < num_examples; i++) {
        if (fscanf(fp, "%d", &y[i]) != 1) {
            fprintf(stderr, "Erreur lecture y à la ligne %d dans %s\n", i+1, filename);
            fclose(fp);
            exit(EXIT_FAILURE);
        }
    }
    fclose(fp);
}

/* ---------- Learning-rate helper ---------- */
double lr_value_iter(lr_schedule_t sched, double lr0, double decay,
                int epoch, int step_every, double gamma)
{
    switch (sched) {
        case LR_CONST: return lr0;
        case LR_TIME:  return lr0 / (1.0 + decay * (double)epoch);
        case LR_EXP:   return lr0 * exp(-decay * (double)epoch);
        case LR_STEP: {
            if (step_every <= 0) return lr0;
            int k = epoch / step_every;        // floor
            double g = pow(gamma, (double)k);
            return lr0 * g;
        }
        default:       return lr0;
    }
}
