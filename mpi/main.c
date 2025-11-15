#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#include "model.h"
#include "utils.h"

#ifdef USE_MPI
#include <mpi.h>
#endif

typedef struct {
    int    hdim;
    int    epochs;
    double lr0;
    double reg;
    int    print_every;
    int    batch_size;
    int    activation;    // act_t
    double lr_decay;      // time-based: lr = lr0 / (1 + decay * epoch)
    int    tasks_mode;    // kept for CLI parity (not used in MPI logic)
} Args;

static void parse_args(int argc, char **argv, Args *a) {
    if (argc < 11) {
        fprintf(stderr,
          "Usage: %s HDIM EPOCHS LR REG PRINT_EVERY BATCH_SIZE DTYPE LR_DECAY ACT TASKS\n",
          argv[0]);
        fprintf(stderr,
          "Example: %s 128 2000 0.01 0.01 1000 32 1 0.0001 3 0\n", argv[0]);
        exit(1);
    }
    a->hdim        = atoi(argv[1]);
    a->epochs      = atoi(argv[2]);
    a->lr0         = atof(argv[3]);
    a->reg         = atof(argv[4]);
    a->print_every = atoi(argv[5]);
    a->batch_size  = atoi(argv[6]);
    (void)argv[7]; // DTYPE unused (compat with OpenMP CLI)
    a->lr_decay    = atof(argv[8]);
    a->activation  = atoi(argv[9]);
    a->tasks_mode  = atoi(argv[10]);
}

int main(int argc, char **argv) {
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif

    int world = 1, rank = 0;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    Args A;
    parse_args(argc, argv, &A);
    const act_t ACT = (act_t)A.activation;

    // Load dataset on every rank: dir = "data" (contains data_X.txt, data_y.txt)
    double *X = NULL;
    int    *y = NULL;
    int N = 0, in_dim = 0, out_dim = 0;
    if (load_dataset("data", &X, &y, &N, &in_dim, &out_dim) != 0) {
        if (rank == 0)
            fprintf(stderr, "[rank %d] Could not load dataset from 'data/'\n", rank);
#ifdef USE_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#else
        return 1;
#endif
    }

    if (rank == 0) {
        printf("Chargement de %d échantillons.\n", N);
#ifdef _OPENMP
        printf("BUILD %s  _OPENMP=%d\n", __DATE__ " " __TIME__, _OPENMP);
#else
        printf("BUILD %s  _OPENMP=OFF\n", __DATE__ " " __TIME__);
#endif
#ifdef USE_MPI
        printf("MPI world size = %d\n", world);
#endif
    }

    MLP m;
    mlp_init(&m, in_dim, A.hdim, out_dim, 42u + (unsigned)rank);

    int *perm = (int*)xmalloc((size_t)N * sizeof(int));
    uint64_t t0 = now_millis();

    int nW1 = in_dim * A.hdim;
    int nb1 = A.hdim;
    int nW2 = A.hdim * out_dim;
    int nb2 = out_dim;
    int G   = nW1 + nb1 + nW2 + nb2;

    double *grad_local  = (double*)xcalloc((size_t)G, sizeof(double));
    double *grad_global = (double*)xcalloc((size_t)G, sizeof(double));

    for (int epoch = 0; epoch < A.epochs; ++epoch) {
        double lr_t = A.lr0 / (1.0 + A.lr_decay * (double)epoch);

        if (rank == 0)
            make_permutation(perm, N, (unsigned)(1234 + epoch));
#ifdef USE_MPI
        MPI_Bcast(perm, N, MPI_INT, 0, MPI_COMM_WORLD);
#endif

        for (int start = 0; start < N; start += A.batch_size) {
            int bcount = A.batch_size;
            if (start + bcount > N) bcount = N - start;

            int lo = start + (bcount * rank) / world;
            int hi = start + (bcount * (rank + 1)) / world;

            memset(grad_local, 0, (size_t)G * sizeof(double));

            double *dW1 = grad_local;
            double *db1 = dW1 + nW1;
            double *dW2 = db1 + nb1;
            double *db2 = dW2 + nW2;

            double local_loss_sum = 0.0;

            for (int t = lo; t < hi; ++t) {
                int idx = perm[t];
                const double *x = &X[idx * in_dim];
                int target = y[idx];
                mlp_accum_grad_one(&m, x, target, ACT,
                                   dW1, db1, dW2, db2,
                                   &local_loss_sum);
            }

#ifdef USE_MPI
            MPI_Allreduce(grad_local, grad_global, G, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double global_loss_sum = 0.0;
            MPI_Allreduce(&local_loss_sum, &global_loss_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
            memcpy(grad_global, grad_local, (size_t)G * sizeof(double));
            double global_loss_sum = local_loss_sum;
#endif

            double *gW1 = grad_global;
            double *gb1 = gW1 + nW1;
            double *gW2 = gb1 + nb1;
            double *gb2 = gW2 + nW2;

            mlp_apply_update(&m, gW1, gb1, gW2, gb2, lr_t, A.reg, bcount);
        }

        if (A.print_every > 0 && (epoch % A.print_every == 0 || epoch == A.epochs - 1)) {
            int e_lo = (N * rank)     / world;
            int e_hi = (N * (rank+1)) / world;
            long long local_correct = 0;
            double local_loss = 0.0;

            mlp_eval_range(&m, X, y, in_dim, out_dim, e_lo, e_hi,
                           ACT, &local_correct, &local_loss);

            long long glob_corr = local_correct;
            double glob_loss = local_loss;
#ifdef USE_MPI
            MPI_Allreduce(&local_correct, &glob_corr, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            MPI_Allreduce(&local_loss, &glob_loss, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
            if (rank == 0) {
                double acc = (double)glob_corr / (double)N;
                printf("Epoch %d, Loss: %.6f, LR: %.6f, Acc: %.4f\n",
                       epoch, glob_loss / (double)N, lr_t, acc);
                fflush(stdout);
            }
        }
    }

    uint64_t t1 = now_millis();
    if (rank == 0) {
        printf("TIME_MS: %.3f (procs=%d)\n", (double)(t1 - t0), world);
    }

    free(grad_local);
    free(grad_global);
    free(perm);
    mlp_free(&m);
    free(X);
    free(y);

#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
