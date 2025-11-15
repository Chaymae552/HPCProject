// main.c (OpenMP version) — const/time/exp/step schedules + robust CLI

#include "model.h"
#include "utils.h"     // brings lr_schedule_t and lr_value() declaration
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <locale.h>
#include <strings.h>   // strcasecmp (POSIX)

#ifdef _WIN32
  #include <direct.h>
  #define MAKE_DIR(p) _mkdir(p)
#else
  #include <sys/stat.h>
  #define MAKE_DIR(p) mkdir(p, 0777)
#endif

#ifdef _OPENMP
  #include <omp.h>
#endif

/* step schedule globals live in utils.c; declare here to set from CLI */
extern int    step_every;   // epochs between steps
extern double step_gamma;   // multiplicative decay at each step

/* small helpers */
static int file_exists(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static void usage(const char *prog) {
    fprintf(stderr,
      "Usage (legacy positional, only if NO flags are used):\n"
      "  %s [hidden] [passes] [lr0] [reg] [print_every] [batch]\n"
      "     [decay_type] [decay_rate] [act] [use_tasks]\n"
      "    decay_type: 0=const, 1=time, 2=exp, 3=step\n"
      "    decay_rate: k for time/exp; for step use --step-gamma\n"
      "\n"
      "Flags (override positional):\n"
      "  --sched|--lr-schedule {const|time|exp|step}\n"
      "  --decay <float>             (k for time/exp)\n"
      "  --step-size <int>           (epochs between steps, step schedule)\n"
      "  --step-gamma <float>        (gamma for step schedule)\n"
      "  --data-dir <path>           (default: data)\n"
      "  --print-every <int>         (e.g., 100 or 1000)\n"
      "  --act {tanh|relu|sigmoid|leaky}\n",
      prog);
}

static lr_schedule_t parse_sched_str(const char *s, lr_schedule_t def) {
    if (!s) return def;
    if (!strcasecmp(s, "const")) return LR_CONST;
    if (!strcasecmp(s, "time"))  return LR_TIME;
    if (!strcasecmp(s, "exp"))   return LR_EXP;
    if (!strcasecmp(s, "step"))  return LR_STEP;
    return def;
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");

    /* dataset paths */
    const char *data_dir = "data";
    char file_X[512], file_y[512];
    snprintf(file_X, sizeof(file_X), "%s/data_X.txt", data_dir);
    snprintf(file_y, sizeof(file_y), "%s/data_y.txt", data_dir);

    /* defaults (globals expected by model.c) */
    nn_input_dim  = 2;
    nn_output_dim = 2;

    int   nn_hdim     = 10;
    int   num_passes  = 20000;
    int   print_every = 1000;     /* this is your old print_loss cadence */

    epsilon         = 0.01;       /* lr0 */
    reg_lambda      = 0.01;
    batch_size      = 32;
    decay_type      = LR_TIME;    /* default schedule */
    decay_rate      = 1e-4;       /* k */
    activation_type = 3;          /* leaky */
    use_omp_tasks   = 0;

    /* step schedule defaults */
    step_every = 500;
    step_gamma = 0.5;

    /* quick help */
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        usage(argv[0]);
        return 0;
    }

    /* legacy positional parsing: only if no flags are present */
    int saw_flag = 0;
    for (int i = 1; i < argc; ++i) {
        if (argv[i][0] == '-') { saw_flag = 1; break; }
    }
    if (!saw_flag && argc > 1) {
        if (argc > 1)  nn_hdim       = atoi(argv[1]);
        if (argc > 2)  num_passes    = atoi(argv[2]);
        if (argc > 3)  epsilon       = atof(argv[3]);        /* lr0 */
        if (argc > 4)  reg_lambda    = atof(argv[4]);
        if (argc > 5)  print_every   = atoi(argv[5]);
        if (argc > 6)  batch_size    = atoi(argv[6]);
        if (argc > 7)  decay_type    = atoi(argv[7]);
        if (argc > 8)  decay_rate    = atof(argv[8]);        /* k (or gamma if you insist) */
        if (argc > 9)  activation_type = atoi(argv[9]);
        if (argc > 10) use_omp_tasks = atoi(argv[10]);
    }

    /* flag parsing (overrides positional) */
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }

        else if ((!strcmp(argv[i], "--lr-schedule") || !strcmp(argv[i], "--sched")) && i+1 < argc) {
            lr_schedule_t s = parse_sched_str(argv[i+1], (lr_schedule_t)decay_type);
            decay_type = (int)s; i++;

        } else if (!strcmp(argv[i], "--decay") && i+1 < argc) {
            decay_rate = atof(argv[++i]);                   /* k for time/exp */

        } else if (!strcmp(argv[i], "--step-size") && i+1 < argc) {
            step_every = atoi(argv[++i]);

        } else if (!strcmp(argv[i], "--step-gamma") && i+1 < argc) {
            step_gamma = atof(argv[++i]);

        } else if (!strcmp(argv[i], "--data-dir") && i+1 < argc) {
            data_dir = argv[++i];
            snprintf(file_X, sizeof(file_X), "%s/data_X.txt", data_dir);
            snprintf(file_y, sizeof(file_y), "%s/data_y.txt", data_dir);

        } else if (!strcmp(argv[i], "--print-every") && i+1 < argc) {
            print_every = atoi(argv[++i]);

        } else if (!strcmp(argv[i], "--act") && i+1 < argc) {
            const char *a = argv[++i];
            if      (!strcasecmp(a,"tanh"))    activation_type = 0;
            else if (!strcasecmp(a,"relu"))    activation_type = 1;
            else if (!strcasecmp(a,"sigmoid")) activation_type = 2;
            else if (!strcasecmp(a,"leaky"))   activation_type = 3;
        }
        else if (!strcmp(argv[i], "--epochs") && i+1 < argc) {
            num_passes = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--hidden") && i+1 < argc) {
            nn_hdim = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--batch") && i+1 < argc) {
            batch_size = atoi(argv[++i]);
        }

    }

    /* dataset checks */
    if (!file_exists(file_X) || !file_exists(file_y)) {
        fprintf(stderr,
            "Erreur: fichiers de données introuvables.\n"
            "  Recherché: '%s' et '%s'\n"
            "Astuce:\n"
            "  mkdir -p %s\n"
            "  python3 generate_moon.py --n-samples 10000 --outdir %s\n",
            file_X, file_y, data_dir, data_dir);
        return 1;
    }

    /* load data */
    num_examples = count_lines(file_y);
    if (num_examples <= 0) {
        fprintf(stderr, "Erreur: nombre d'échantillons invalide (%d) dans %s\n",
                num_examples, file_y);
        return 1;
    }
    printf("Chargement de %d échantillons.\n", num_examples);

    X = (double*) xmalloc((size_t)num_examples * nn_input_dim * sizeof(double));
    y = (int*)    xmalloc((size_t)num_examples * sizeof(int));
    load_X(file_X, X, num_examples, nn_input_dim);
    load_y(file_y, y, num_examples);

    /* build info + timer */
#ifdef _OPENMP
    printf("BUILD %s  _OPENMP=%d\n", __DATE__ " " __TIME__, _OPENMP);
    printf("OMP_NUM_THREADS=%d (max=%d)\n", omp_get_max_threads(), omp_get_max_threads());
    double t0 = omp_get_wtime();
#else
    printf("BUILD %s  (OpenMP disabled)\n", __DATE__ " " __TIME__);
    double t0 = 0.0;
#endif

    (void)MAKE_DIR("output");

    /* banner (show effective schedule params) */
    const char *sched_name =
        (decay_type==LR_CONST) ? "const" :
        (decay_type==LR_TIME ) ? "time"  :
        (decay_type==LR_EXP  ) ? "exp"   : "step";

    if (decay_type == LR_CONST) {
        printf("LR schedule: %s  (lr0=%.6f)\n", sched_name, epsilon);
    } else if (decay_type == LR_TIME) {
        printf("LR schedule: %s  (lr0=%.6f, k=%.6f)  lr=lr0/(1+k*t)\n",
               sched_name, epsilon, decay_rate);
    } else if (decay_type == LR_EXP) {
        printf("LR schedule: %s  (lr0=%.6f, k=%.6f)  lr=lr0*exp(-k*t)\n",
               sched_name, epsilon, decay_rate);
    } else { /* step */
        printf("LR schedule: %s  (lr0=%.6f, step_size=%d, gamma=%.6f)\n",
               sched_name, epsilon, step_every, step_gamma);
    }

    /* train */
    /* map our print cadence into the existing API param name expected by model.c */
    int print_loss = (print_every > 0) ? print_every : 0;
    build_model(nn_hdim, num_passes, print_loss);

    /* timing */
#ifdef _OPENMP
    double t1 = omp_get_wtime();
    printf("TIME_MS: %.3f (threads=%d, tasks=%d)\n",
           1000.0*(t1 - t0), omp_get_max_threads(), use_omp_tasks);
#endif

    /* cleanup */
    free(X); X = NULL;
    free(y); y = NULL;
    return 0;
}
