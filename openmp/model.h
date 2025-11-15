#ifndef MODEL_H
#define MODEL_H

#ifdef __cplusplus
extern "C" {
#endif

extern int    num_examples;
extern int    nn_input_dim;
extern int    nn_output_dim;

extern double reg_lambda;    
extern double epsilon;       

extern double *X;            
extern int    *y;         

// -------------------- Training hyperparameters (configurable) --------------------
// Learning-rate schedule:
//   0=const, 1=time-based, 2=exp, 3=step
extern double initial_lr;     // η0
extern double decay_rate;     // k
extern int    decay_type;     // 0=const, 1=time-based, 2=exp, 3=step

// Step schedule params 
extern int    step_every;     // epochs per step (<=0 disables step behavior)
extern double step_gamma;     // multiplicative factor per step 

// Activation
//   0=tanh, 1=ReLU, 2=Sigmoid, 3=LeakyReLU
extern int    activation_type;
extern double leaky_alpha;    // for LeakyReLU

// Mini-batch & parallelism
extern int    batch_size;     // mini-batch size
extern int    use_omp_tasks;  // 0=normal SGD; 1=task-parallel epoch reductions

double calculate_loss(double *W1, double *b1, double *W2, double *b2, int nn_hdim);
void   build_model(int nn_hdim, int num_passes, int print_loss);

// -------------------- Optional helpers --------------------
int    predict_one(double *x,
                   double *W1, double *b1,
                   double *W2, double *b2,
                   int in_dim, int hdim, int out_dim);

double accuracy(double *X_, int *y_, int N,
                double *W1, double *b1,
                double *W2, double *b2,
                int in_dim, int hdim, int out_dim);

#ifdef __cplusplus
}
#endif

#endif
