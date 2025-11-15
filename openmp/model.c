// model.c — MINI-BATCH GD + CSV (loss, lr, acc)
//           Dynamic LR (const/time/exp/step)
//           Selectable Activations (tanh/ReLU/Sigmoid/Leaky)
//           OpenMP parallelization of kernels and grads
//           Optional OpenMP TASKS mode: parallelize batches within an epoch

#include "model.h"
#include "utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#ifdef _OPENMP
  #include <omp.h>
#endif
#if !defined(_WIN32)
  #include <alloca.h>   // for alloca on Linux/glibc
#endif

#ifdef _WIN32
  #include <direct.h>
  #define MAKE_DIR(p) _mkdir(p)
#else
  #include <sys/stat.h>
  #define MAKE_DIR(p) mkdir(p, 0777)
#endif

#ifndef SAFE_FREE
#define SAFE_FREE(p) do { if ((p)!=NULL) { free(p); (p)=NULL; } } while (0)
#endif

/* =======================================================================
   Project-wide globals (set/printed in main.c; used here in training)
   ======================================================================= */
int    num_examples     = 0;
int    nn_input_dim     = 0;
int    nn_output_dim    = 0;

double reg_lambda       = 0.01;
double epsilon          = 0.01;   // CLI hook

/* Learning-rate schedule */
double initial_lr       = 0.01;   // lr0 (we tie to epsilon at build_model start)
double decay_rate       = 0.0001; // k (for time/exp)
int    decay_type       = 1;      // 0=const, 1=time, 2=exp, 3=step

/* Step-schedule parameters (overridden by flags in main.c) */
int    step_every       = 0;      // epochs between steps (0 disables step behavior)
double step_gamma       = 0.5;    // multiplicative factor at each step

/* Activations */
int    activation_type  = 3;      // 0=tanh,1=ReLU,2=Sigmoid,3=Leaky
double leaky_alpha      = 0.01;

/* Mini-batch + Tasks */
int    batch_size       = 32;
int    use_omp_tasks    = 0;      // 0=SGD per-batch; 1=task-epoch reduction

/* Dataset pointers (filled by main.c) */
double *X = NULL; // (num_examples × nn_input_dim)
int    *y = NULL; // (num_examples)

/* =======================================================================
   Helpers: activations / schedule / names / timing
   ======================================================================= */

static inline double act_fn(double x, int type) {
    switch (type) {
        case 1:  return (x > 0.0) ? x : 0.0;                    // ReLU
        case 2:  return 1.0 / (1.0 + exp(-x));                  // Sigmoid
        case 3:  return (x > 0.0) ? x : leaky_alpha * x;        // Leaky ReLU
        default: return tanh(x);                                // Tanh
    }
}

static inline double act_grad_from_a(double a, double x, int type) {
    (void)a; // a is useful for sigmoid/tanh if you cache it; we keep both forms
    switch (type) {
        case 1:  return (x > 0.0) ? 1.0 : 0.0;                  // ReLU'
        case 2:  return a * (1.0 - a);                          // Sigmoid'
        case 3:  return (x > 0.0) ? 1.0 : leaky_alpha;          // Leaky'
        default: return 1.0 - a*a;                              // Tanh'
    }
}

static const char* act_name(void){
    switch (activation_type){
        case 1: return "relu";
        case 2: return "sigmoid";
        case 3: return "leaky";
        default: return "tanh";
    }
}
static const char* decay_name(void){
    switch (decay_type){
        case 1: return "timebased";
        case 2: return "exp";
        case 3: return "step";
        default: return "const";
    }
}

/* Simple wall-clock timer for performance measurements */
static inline double wall_time(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return (double)clock() / (double)CLOCKS_PER_SEC;
#endif
}

/* Learning-rate value per optimizer step (iteration)
   t = iteration counter (we update per mini-batch in SGD mode; once per epoch in TASKS mode) */
static inline double lr_value(long long t) {
    /* const */
    if (decay_type == 0) return initial_lr;

    /* time-based: lr_t = lr0 / (1 + k * t) */
    if (decay_type == 1) return initial_lr / (1.0 + decay_rate * (double)t);

    /* exponential: lr_t = lr0 * exp(-k * t) */
    if (decay_type == 2) return initial_lr * exp(-decay_rate * (double)t);

    /* step-based: lr_t = lr0 * gamma^(floor(t / step_every)) */
    if (decay_type == 3 && step_every > 0) {
        long long s = t / (long long)step_every;
        /* pow with integer exponent; cast to double for pow */
        return initial_lr * pow(step_gamma, (double)s);
    }

    /* fallback */
    return initial_lr;
}

/* =======================================================================
   Full-dataset loss (NLL + L2)
   ======================================================================= */
double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim)
{
    double *z1    = (double*)xcalloc((size_t)num_examples * nn_hdim,       sizeof(double));
    double *a1    = (double*)xcalloc((size_t)num_examples * nn_hdim,       sizeof(double));
    double *z2    = (double*)xcalloc((size_t)num_examples * nn_output_dim, sizeof(double));
    double *probs = (double*)xcalloc((size_t)num_examples * nn_output_dim, sizeof(double));

    matmul(X, W1, z1, num_examples, nn_input_dim, nn_hdim);
    add_bias(z1, b1, num_examples, nn_hdim);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < num_examples * nn_hdim; i++)
        a1[i] = act_fn(z1[i], activation_type);

    matmul(a1, W2, z2, num_examples, nn_hdim, nn_output_dim);
    add_bias(z2, b2, num_examples, nn_output_dim);
    softmax(z2, probs, num_examples, nn_output_dim);

    double data_loss = 0.0;
    for (int i = 0; i < num_examples; i++) {
        int label = y[i];
        double p = probs[i*nn_output_dim + label];
        data_loss += -log(p);
    }
    /* L2 */
    double reg = 0.0;
    for (int i = 0; i < nn_input_dim * nn_hdim; i++) reg += W1[i]*W1[i];
    for (int i = 0; i < nn_hdim * nn_output_dim; i++) reg += W2[i]*W2[i];
    data_loss += reg_lambda / 2.0 * reg;

    double loss = data_loss / (double)num_examples;

    SAFE_FREE(z1); SAFE_FREE(a1); SAFE_FREE(z2); SAFE_FREE(probs);
    return loss;
}

/* =======================================================================
   Accuracy helpers
   ======================================================================= */
int predict_one(double *x,
                double *W1, double *b1,
                double *W2, double *b2,
                int in_dim, int hdim, int out_dim)
{
    double *z1 = (double*)alloca((size_t)hdim * sizeof(double));
    double *a1 = (double*)alloca((size_t)hdim * sizeof(double));
    double *z2 = (double*)alloca((size_t)out_dim * sizeof(double));

    /* z1 = x * W1 + b1 */
    for (int j = 0; j < hdim; ++j) {
        double sum = b1[j];
        for (int i = 0; i < in_dim; ++i)
            sum += x[i] * W1[i*hdim + j];
        z1[j] = sum;
        a1[j] = act_fn(sum, activation_type);
    }
    /* z2 = a1 * W2 + b2 */
    for (int k = 0; k < out_dim; ++k) {
        double sum = b2[k];
        for (int j = 0; j < hdim; ++j)
            sum += a1[j] * W2[j*out_dim + k];
        z2[k] = sum;
    }
    /* argmax */
    int argmax = 0;
    double best = z2[0];
    for (int k = 1; k < out_dim; ++k)
        if (z2[k] > best) { best = z2[k]; argmax = k; }
    return argmax;
}

double accuracy(double *X_, int *y_, int N,
                double *W1, double *b1,
                double *W2, double *b2,
                int in_dim, int hdim, int out_dim)
{
    int correct = 0;
    #pragma omp parallel for reduction(+:correct) schedule(static)
    for (int n = 0; n < N; ++n) {
        int pred = predict_one(&X_[n*in_dim], W1, b1, W2, b2, in_dim, hdim, out_dim);
        if (pred == y_[n]) correct++;
    }
    return (double)correct / (double)N;
}

/* =======================================================================
   build_model — SGD (default) or Task-parallel epoch reductions
   ======================================================================= */
void build_model(int nn_hdim, int num_passes, int print_loss)
{
    srand(0);

    /* Tie CLI epsilon to schedule's initial_lr (if user didn't set initial_lr explicitly) */
    if (!(initial_lr > 0.0)) initial_lr = epsilon;

    /* Start timer for performance measurement */
    double t0 = wall_time();

    (void)MAKE_DIR("output");
    char csv_path[256];
    snprintf(csv_path, sizeof(csv_path), "output/%s_%s.csv", act_name(), decay_name());
    FILE *flog = fopen(csv_path, "w");
    if (flog) fprintf(flog, "epoch,loss,lr,accuracy\n");
    printf("Logging to %s\n", csv_path);

    const int num_batches = (num_examples + batch_size - 1) / batch_size;

    /* Parameters */
    double *W1 = (double*)xmalloc((size_t)nn_input_dim * nn_hdim       * sizeof(double));
    double *b1 = (double*)xcalloc(                 nn_hdim,            sizeof(double));
    double *W2 = (double*)xmalloc((size_t)nn_hdim  * nn_output_dim     * sizeof(double));
    double *b2 = (double*)xcalloc(                 nn_output_dim,      sizeof(double));

    /* Xavier-ish init */
    for (int i = 0; i < nn_input_dim * nn_hdim; i++)
        W1[i] = randn() / sqrt((double)nn_input_dim);
    for (int i = 0; i < nn_hdim * nn_output_dim; i++)
        W2[i] = randn() / sqrt((double)nn_hdim);

    /* Reusable buffers for SGD mode (max batch) */
    double *z1    = (double*)xcalloc((size_t)batch_size * nn_hdim,       sizeof(double));
    double *a1    = (double*)xcalloc((size_t)batch_size * nn_hdim,       sizeof(double));
    double *z2    = (double*)xcalloc((size_t)batch_size * nn_output_dim, sizeof(double));
    double *probs = (double*)xcalloc((size_t)batch_size * nn_output_dim, sizeof(double));

    double *delta3= (double*)xmalloc ((size_t)batch_size * nn_output_dim * sizeof(double));
    double *dW2   = (double*)xcalloc((size_t)nn_hdim    * nn_output_dim,  sizeof(double));
    double *db2   = (double*)xcalloc(                 nn_output_dim,      sizeof(double));
    double *delta2= (double*)xcalloc((size_t)batch_size * nn_hdim,        sizeof(double));
    double *dW1   = (double*)xcalloc((size_t)nn_input_dim * nn_hdim,      sizeof(double));
    double *db1g  = (double*)xcalloc(                 nn_hdim,            sizeof(double));

    long long iter = 0; /* optimizer step counter for lr_value() */

    for (int epoch = 0; epoch < num_passes; epoch++) {

        if (!use_omp_tasks) {
            /* =========================================================
               (A) Standard SGD: update per-batch
               ========================================================= */
            for (int b = 0; b < num_batches; b++) {
                const int start = b * batch_size;
                const int end   = (start + batch_size < num_examples) ? (start + batch_size) : num_examples;
                const int bs    = end - start;

                double *Xb = &X[start * nn_input_dim];
                int    *yb = &y[start];

                /* reset batch buffers / grads */
                memset(z1,    0, (size_t)bs * nn_hdim        * sizeof(double));
                memset(a1,    0, (size_t)bs * nn_hdim        * sizeof(double));
                memset(z2,    0, (size_t)bs * nn_output_dim  * sizeof(double));
                memset(probs, 0, (size_t)bs * nn_output_dim  * sizeof(double));
                memset(delta2,0, (size_t)bs * nn_hdim        * sizeof(double));
                memset(delta3,0, (size_t)bs * nn_output_dim  * sizeof(double));
                memset(dW1,   0, (size_t)nn_input_dim * nn_hdim * sizeof(double));
                memset(dW2,   0, (size_t)nn_hdim * nn_output_dim * sizeof(double));
                memset(db1g,  0, (size_t)nn_hdim              * sizeof(double));
                memset(db2,   0, (size_t)nn_output_dim        * sizeof(double));

                /* Forward (batch) */
                matmul(Xb, W1, z1, bs, nn_input_dim, nn_hdim);
                add_bias(z1, b1, bs, nn_hdim);

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < bs * nn_hdim; i++)
                    a1[i] = act_fn(z1[i], activation_type);

                matmul(a1, W2, z2, bs, nn_hdim, nn_output_dim);
                add_bias(z2, b2, bs, nn_output_dim);
                softmax(z2, probs, bs, nn_output_dim);

                /* Backprop (batch) */
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < bs * nn_output_dim; i++)
                    delta3[i] = probs[i];

                #pragma omp parallel for schedule(static)
                for (int n = 0; n < bs; n++)
                    delta3[n*nn_output_dim + yb[n]] -= 1.0;

                /* dW2 = a1^T * delta3 */
                #pragma omp parallel for collapse(2) schedule(static)
                for (int j = 0; j < nn_hdim; j++) {
                    for (int k = 0; k < nn_output_dim; k++) {
                        double sum = 0.0;
                        for (int n = 0; n < bs; n++)
                            sum += a1[n*nn_hdim + j] * delta3[n*nn_output_dim + k];
                        dW2[j*nn_output_dim + k] = sum;
                    }
                }

                /* db2 */
                #pragma omp parallel for schedule(static)
                for (int k = 0; k < nn_output_dim; k++) {
                    double sum = 0.0;
                    for (int n = 0; n < bs; n++) sum += delta3[n*nn_output_dim + k];
                    db2[k] = sum;
                }

                /* delta2 */
                #pragma omp parallel for collapse(2) schedule(static)
                for (int n = 0; n < bs; n++) {
                    for (int j = 0; j < nn_hdim; j++) {
                        double sum = 0.0;
                        for (int k = 0; k < nn_output_dim; k++)
                            sum += delta3[n*nn_output_dim + k] * W2[j*nn_output_dim + k];
                        double gprime = act_grad_from_a(a1[n*nn_hdim + j], z1[n*nn_hdim + j], activation_type);
                        delta2[n*nn_hdim + j] = sum * gprime;
                    }
                }

                /* dW1 = Xb^T * delta2 */
                #pragma omp parallel for collapse(2) schedule(static)
                for (int i = 0; i < nn_input_dim; i++) {
                    for (int j = 0; j < nn_hdim; j++) {
                        double sum = 0.0;
                        for (int n = 0; n < bs; n++)
                            sum += Xb[n*nn_input_dim + i] * delta2[n*nn_hdim + j];
                        dW1[i*nn_hdim + j] = sum;
                    }
                }

                /* db1 */
                #pragma omp parallel for schedule(static)
                for (int j = 0; j < nn_hdim; j++) {
                    double sum = 0.0;
                    for (int n = 0; n < bs; n++) sum += delta2[n*nn_hdim + j];
                    db1g[j] = sum;
                }

                /* L2 regularization */
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nn_hdim * nn_output_dim; i++) dW2[i] += reg_lambda * W2[i];
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nn_input_dim * nn_hdim; i++)   dW1[i] += reg_lambda * W1[i];

                /* LR + Update */
                double lr = lr_value(iter);
                const double inv_bs = 1.0 / (double)bs;

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nn_input_dim * nn_hdim; i++)  W1[i] -= lr * dW1[i] * inv_bs;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nn_hdim; i++)                 b1[i] -= lr * db1g[i] * inv_bs;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nn_hdim * nn_output_dim; i++) W2[i] -= lr * dW2[i] * inv_bs;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < nn_output_dim; i++)           b2[i] -= lr * db2[i] * inv_bs;

                iter++; /* next optimizer step */
            }
        } else {
            /* =========================================================
               (B) OpenMP TASKS mode:
               Compute grads in parallel across all batches, then single update
               ========================================================= */
            const int W1sz = nn_input_dim * nn_hdim;
            const int W2sz = nn_hdim * nn_output_dim;

            int T = 1;
            #ifdef _OPENMP
            T = omp_get_max_threads();
            #endif

            /* Task granularity control: only create a task if bs >= OMP_TASK_MIN_BS (default 1) */
            int task_min_bs = 1;
            const char *env_min_bs = getenv("OMP_TASK_MIN_BS");
            if (env_min_bs) {
                int tmp = atoi(env_min_bs);
                if (tmp > 0) task_min_bs = tmp;
            }

            double *dW1_all = (double*)xcalloc((size_t)T * W1sz, sizeof(double));
            double *dW2_all = (double*)xcalloc((size_t)T * W2sz, sizeof(double));
            double *db1_all = (double*)xcalloc((size_t)T * nn_hdim, sizeof(double));
            double *db2_all = (double*)xcalloc((size_t)T * nn_output_dim, sizeof(double));

            #pragma omp parallel
            {
                #pragma omp single nowait
                for (int b = 0; b < num_batches; b++) {
                    const int start = b * batch_size;
                    const int end   = (start + batch_size < num_examples) ? (start + batch_size) : num_examples;
                    const int bs    = end - start;

                    #pragma omp task firstprivate(start,end,bs) shared(dW1_all,dW2_all,db1_all,db2_all) if (bs >= task_min_bs)
                    {
                        const double *Xb = &X[start * nn_input_dim];
                        const int    *yb = &y[start];

                        /* Local temporaries & local grads */
                        double *lz1    = (double*)xcalloc((size_t)bs * nn_hdim,       sizeof(double));
                        double *la1    = (double*)xcalloc((size_t)bs * nn_hdim,       sizeof(double));
                        double *lz2    = (double*)xcalloc((size_t)bs * nn_output_dim, sizeof(double));
                        double *lprobs = (double*)xcalloc((size_t)bs * nn_output_dim, sizeof(double));
                        double *ld3    = (double*)xmalloc ((size_t)bs * nn_output_dim * sizeof(double));
                        double *ld2    = (double*)xcalloc((size_t)bs * nn_hdim,       sizeof(double));
                        double *lW1    = (double*)xcalloc((size_t)W1sz, sizeof(double));
                        double *lW2    = (double*)xcalloc((size_t)W2sz, sizeof(double));
                        double *lb1    = (double*)xcalloc((size_t)nn_hdim,       sizeof(double));
                        double *lb2    = (double*)xcalloc((size_t)nn_output_dim,   sizeof(double));

                        /* Forward */
                        matmul((double*)Xb, W1, lz1, bs, nn_input_dim, nn_hdim);
                        add_bias(lz1, b1, bs, nn_hdim);
                        for (int i = 0; i < bs * nn_hdim; i++)
                            la1[i] = act_fn(lz1[i], activation_type);
                        matmul(la1, W2, lz2, bs, nn_hdim, nn_output_dim);
                        add_bias(lz2, b2, bs, nn_output_dim);
                        softmax(lz2, lprobs, bs, nn_output_dim);

                        /* Backprop */
                        for (int i = 0; i < bs * nn_output_dim; i++) ld3[i] = lprobs[i];
                        for (int n = 0; n < bs; n++)
                            ld3[n*nn_output_dim + yb[n]] -= 1.0;

                        /* lW2, lb2 */
                        for (int j = 0; j < nn_hdim; j++) {
                            for (int k = 0; k < nn_output_dim; k++) {
                                double sum = 0.0;
                                for (int n = 0; n < bs; n++)
                                    sum += la1[n*nn_hdim + j] * ld3[n*nn_output_dim + k];
                                lW2[j*nn_output_dim + k] = sum;
                            }
                        }
                        for (int k = 0; k < nn_output_dim; k++) {
                            double sum = 0.0;
                            for (int n = 0; n < bs; n++) sum += ld3[n*nn_output_dim + k];
                            lb2[k] = sum;
                        }

                        /* ld2 */
                        for (int n = 0; n < bs; n++) {
                            for (int j = 0; j < nn_hdim; j++) {
                                double sum = 0.0;
                                for (int k = 0; k < nn_output_dim; k++)
                                    sum += ld3[n*nn_output_dim + k] * W2[j*nn_output_dim + k];
                                double gprime = act_grad_from_a(la1[n*nn_hdim + j], lz1[n*nn_hdim + j], activation_type);
                                ld2[n*nn_hdim + j] = sum * gprime;
                            }
                        }

                        /* lW1, lb1 */
                        for (int i = 0; i < nn_input_dim; i++) {
                            for (int j = 0; j < nn_hdim; j++) {
                                double sum = 0.0;
                                for (int n = 0; n < bs; n++)
                                    sum += Xb[n*nn_input_dim + i] * ld2[n*nn_hdim + j];
                                lW1[i*nn_hdim + j] = sum;
                            }
                        }
                        for (int j = 0; j < nn_hdim; j++) {
                            double sum = 0.0;
                            for (int n = 0; n < bs; n++) sum += ld2[n*nn_hdim + j];
                            lb1[j] = sum;
                        }

                        /* Thread-private accumulation */
                        int tid = 0;
                        #ifdef _OPENMP
                        tid = omp_get_thread_num();
                        #endif
                        const size_t o1 = (size_t)tid * (size_t)W1sz;
                        const size_t o2 = (size_t)tid * (size_t)W2sz;
                        const size_t ob1= (size_t)tid * (size_t)nn_hdim;
                        const size_t ob2= (size_t)tid * (size_t)nn_output_dim;

                        for (int i = 0; i < W1sz; i++) dW1_all[o1 + i] += lW1[i];
                        for (int i = 0; i < W2sz; i++) dW2_all[o2 + i] += lW2[i];
                        for (int i = 0; i < nn_hdim; i++) db1_all[ob1 + i] += lb1[i];
                        for (int i = 0; i < nn_output_dim; i++) db2_all[ob2 + i] += lb2[i];

                        /* Free locals */
                        SAFE_FREE(lz1); SAFE_FREE(la1); SAFE_FREE(lz2); SAFE_FREE(lprobs);
                        SAFE_FREE(ld3); SAFE_FREE(ld2);
                        SAFE_FREE(lW1); SAFE_FREE(lW2); SAFE_FREE(lb1); SAFE_FREE(lb2);
                    } // task
                } // for batches
            } // parallel (implicit taskwait)

            /* Reduce across threads */
            memset(dW1, 0, (size_t)nn_input_dim * nn_hdim * sizeof(double));
            memset(dW2, 0, (size_t)nn_hdim * nn_output_dim * sizeof(double));
            memset(db1g, 0, (size_t)nn_hdim * sizeof(double));
            memset(db2,  0, (size_t)nn_output_dim * sizeof(double));

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_input_dim * nn_hdim; ++i) {
                double sum = 0.0;
                for (int t = 0; t < T; ++t) sum += dW1_all[t*(nn_input_dim*nn_hdim) + i];
                dW1[i] = sum;
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_hdim * nn_output_dim; ++i) {
                double sum = 0.0;
                for (int t = 0; t < T; ++t) sum += dW2_all[t*(nn_hdim*nn_output_dim) + i];
                dW2[i] = sum;
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_hdim; ++i) {
                double sum = 0.0;
                for (int t = 0; t < T; ++t) sum += db1_all[t*nn_hdim + i];
                db1g[i] = sum;
            }
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_output_dim; ++i) {
                double sum = 0.0;
                for (int t = 0; t < T; ++t) sum += db2_all[t*nn_output_dim + i];
                db2[i] = sum;
            }

            /* L2 regularization */
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_hdim * nn_output_dim; i++) dW2[i] += reg_lambda * W2[i];
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)   dW1[i] += reg_lambda * W1[i];

            /* Single update with averaged gradient over full dataset */
            double lr = lr_value(iter);
            const double inv_all = 1.0 / (double)num_examples;

            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_input_dim * nn_hdim; i++)  W1[i] -= lr * dW1[i] * inv_all;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_hdim; i++)                 b1[i] -= lr * db1g[i] * inv_all;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_hdim * nn_output_dim; i++) W2[i] -= lr * dW2[i] * inv_all;
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nn_output_dim; i++)           b2[i] -= lr * db2[i] * inv_all;

            iter++; /* count as one optimizer step */

            SAFE_FREE(dW1_all); SAFE_FREE(dW2_all); SAFE_FREE(db1_all); SAFE_FREE(db2_all);
        }

        /* Logging every 1000 epochs (keep identical to your behavior) */
        if (print_loss && epoch % 1000 == 0) {
            double loss   = calculate_loss(W1, b1, W2, b2, nn_hdim);
            double lr_now = lr_value(iter);
            double acc    = accuracy(X, y, num_examples, W1, b1, W2, b2,
                                     nn_input_dim, nn_hdim, nn_output_dim);

            printf("Epoch %d, Loss: %.6f, LR: %.6f, Acc: %.4f\n",
                   epoch, loss, lr_now, acc);

            if (flog) fprintf(flog, "%d,%.6f,%.6f,%.4f\n", epoch, loss, lr_now, acc);
        }
    }

    /* Stop timer and print performance line */
    double t1 = wall_time();
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("TIME_MS: %.3f (threads=%d, tasks=%d)\n",
           (t1 - t0) * 1000.0, threads, use_omp_tasks);

    /* Save weights (optional for later plotting/inspection) */
    (void)MAKE_DIR("output");
    FILE *fw1 = fopen("output/W1.txt", "w");
    FILE *fb1 = fopen("output/b1.txt", "w");
    FILE *fw2 = fopen("output/W2.txt", "w");
    FILE *fb2 = fopen("output/b2.txt", "w");
    if (fw1 && fb1 && fw2 && fb2) {
        for (int i = 0; i < nn_input_dim * nn_hdim; i++) fprintf(fw1, "%lf\n", W1[i]);
        for (int i = 0; i < nn_hdim; i++)                 fprintf(fb1, "%lf\n", b1[i]);
        for (int i = 0; i < nn_hdim * nn_output_dim; i++) fprintf(fw2, "%lf\n", W2[i]);
        for (int i = 0; i < nn_output_dim; i++)           fprintf(fb2, "%lf\n", b2[i]);
    } else {
        fprintf(stderr, "Erreur ouverture fichier de sortie: %s\n", strerror(errno));
    }
    if (fw1) fclose(fw1);
    if (fb1) fclose(fb1);
    if (fw2) fclose(fw2);
    if (fb2) fclose(fb2);
    if (flog) fclose(flog);

    /* Free */
    SAFE_FREE(z1); SAFE_FREE(a1); SAFE_FREE(z2); SAFE_FREE(probs);
    SAFE_FREE(delta3); SAFE_FREE(dW2); SAFE_FREE(db2);
    SAFE_FREE(delta2); SAFE_FREE(dW1); SAFE_FREE(db1g);
    SAFE_FREE(W1); SAFE_FREE(b1); SAFE_FREE(W2); SAFE_FREE(b2);
}
