
#include "slp.h"
#include <stdlib.h>

static void randomize_sequence(uint8_t *arr, uint8_t size) {
  for (uint8_t i = 0; i < size; ++i) {
    arr[i] = rand() % (size + 1);
  }
}

void slp_init(struct slp_ctx *ctx) {
  if (ctx == NULL) {
    return;
  }
  if (ctx->weights == NULL) {
    return;
  }
  // randomly initialize weights
  for (uint16_t i = 0; i < ctx->num_inputs; ++i) {
    // range [0,1]
    ctx->weights[i] = (float)rand() / RAND_MAX;
  }
  // initialize bias to a random value in range [0,1]
  ctx->bias = (float)rand() / RAND_MAX;
}

void slp_run(struct slp_ctx *ctx, const float *inputs) {
  // calculate cumulative sum of layer 1 inputs and weights + bias
  float l1_sum = 0.0f;
  for (uint16_t i = 0; i < ctx->num_inputs; ++i) {
    l1_sum += inputs[i] * ctx->weights[i];
  }
  l1_sum += ctx->bias;
  // pass through activation function
  ctx->output = ctx->activation(l1_sum);
}

void slp_train(struct slp_ctx *ctx, const struct slp_training_params *params) {
  if (ctx == NULL) {
    return;
  }
  if (ctx->weights == NULL) {
    return;
  }
  if (params == NULL) {
    return;
  }

  // implement stochastic learning algorithm -> update weights after each training sample
  float error = 0.0f;
  for (uint32_t epochs = 0; epochs < params->epochs; ++epochs) {
    // for every epoch shuffle training data: select a random sequence of indices from 0 -> num_samples
    uint8_t batch_sequence[10] = {0};
    randomize_sequence(batch_sequence, params->num_samples);
    for (uint8_t i = 0; i < params->num_samples; ++i) {
      slp_run(ctx, params->inputs[batch_sequence[i]]);
      // calculate error
      error = ctx->output - params->outputs[batch_sequence[i]];
      // run backpropagation
      for (uint16_t i = 0; i < ctx->num_inputs; ++i) {
        ctx->weights[i] = ctx->weights[i] + params->learning_rate * error * params->inputs[batch_sequence[i]][i];
      }
      ctx->bias = ctx->bias + params->learning_rate * error;
    }
  }
}
