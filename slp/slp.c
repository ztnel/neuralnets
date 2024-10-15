
#include "slp.h"
#include <stdio.h>
#include <stdlib.h>

// fisher-yates shuffle algorithm (O(n))
static void fisher_yates_shuffle(uint16_t *arr, const uint16_t size) {
  for (int i = size - 1; i >= 0; --i) {
    uint16_t j = rand() % (size - 1);
    uint16_t tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
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
  if (params->sample_indices == NULL) {
    return;
  }
  FILE *fp;
  fp = fopen("train.csv", "w");
  // implement stochastic learning algorithm -> update weights after each training sample
  float error = 0.0f;
  fprintf(fp, "EPOCH, WEIGHT_1, WEIGHT_2, BIAS, ERROR\n");
  for (uint32_t epoch = 0; epoch < params->epochs; ++epoch) {
    // for every epoch shuffle training data: select a random sequence of indices from 0 -> num_samples
    fisher_yates_shuffle(params->sample_indices, params->num_samples);
    for (uint8_t i = 0; i < params->num_samples; ++i) {
      const uint16_t index = params->sample_indices[i];
      const float *inputs = params->samples[index].inputs;
      const float supervised_output = params->samples[params->sample_indices[i]].output;
      slp_run(ctx, inputs);
      // calculate error
      error = supervised_output - ctx->output;
      // run backpropagation
      for (uint16_t j = 0; j < ctx->num_inputs; ++j) {
        ctx->weights[j] = ctx->weights[j] + (params->learning_rate * error * inputs[j]);
      }
      ctx->bias = ctx->bias + (params->learning_rate * error);
    }
    fprintf(fp, "%u,", epoch);
    printf("%u,", epoch);
    for (uint16_t j = 0; j < ctx->num_inputs; ++j) {
      fprintf(fp, "%.1f,", ctx->weights[j]);
      printf("%.1f,", ctx->weights[j]);
    }
    fprintf(fp, "%.1f,%.1f\n", ctx->bias, error);
    printf("%.1f,%.1f\n", ctx->bias, error);
  }
  fclose(fp);
}
