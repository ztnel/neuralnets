/**
 * A Simple Single-Layer Perceptron
 *
 */

#ifndef __SLP_H__
#define __SLP_H__

#include <stdint.h>

#define SLP_MAX_WEIGHTS (uint16_t)10

struct slp_training_params {
  uint32_t epochs;
  float learning_rate;
  const uint16_t num_samples;
  const float *inputs[4];
  const float *outputs;
};

struct slp_ctx {
  uint16_t num_inputs; // number of weights = number of inputs in SLP
  float *weights;      // provided
  float (*activation)(const float a);
  float output;
  float bias; // one bias for SLP
};

void slp_init(struct slp_ctx *ctx);
void slp_train(struct slp_ctx *ctx, const struct slp_training_params *params);
void slp_run(struct slp_ctx *ctx, const float *inputs);

#endif // __SLP_H__
