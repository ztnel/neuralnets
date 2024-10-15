
#include "slp.h"
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_PERMUTATIONS 4
#define NUM_GATE_INPUTS 2
#define TRAIN_EPOCHS (uint16_t)1000

float step_activation(const float in) {
  return in >= 0 ? 1.0f : 0.0f;
}

float relu_activation(const float in) {
  return in >= 0 ? in : 0.0f;
}

float sigmoid_activation(const float in) {
  return 1 / (1 + exp(-in));
}

static void test_slp_and_gate(void) {
  // define inputs and expected outputs
  const float input_permutations[NUM_PERMUTATIONS][NUM_GATE_INPUTS] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
  const float expected_outputs[NUM_PERMUTATIONS] = {0.0f, 0.0f, 1.0f, 0.0f};
  float weights[NUM_GATE_INPUTS] = {0};

  const struct slp_sample samples[NUM_PERMUTATIONS] = {
      {.inputs = input_permutations[0],
       .output = expected_outputs[0]},
      {.inputs = input_permutations[1],
       .output = expected_outputs[1]},
      {.inputs = input_permutations[2],
       .output = expected_outputs[2]},
      {.inputs = input_permutations[3],
       .output = expected_outputs[3]},
  };
  uint16_t batch_sequence[NUM_PERMUTATIONS] = {0, 1, 2, 3};
  // train the slp
  struct slp_ctx slp = {
      .num_inputs = NUM_GATE_INPUTS,
      .activation = step_activation,
      .weights = weights,
  };
  struct slp_training_params params = {
      .epochs = 100,
      .learning_rate = 0.01,
      .num_samples = NUM_PERMUTATIONS,
      .samples = samples,
      .sample_indices = batch_sequence,
  };
  slp_init(&slp);
  slp_train(&slp, &params);
  printf("Training complete. Final Weights: %.2f %.2f Bias: %.2f\n", weights[0], weights[1], slp.bias);

  // run slp over each gate input permutation to verify results
  printf(" --- Verification ---\n");
  for (uint8_t i = 0; i < NUM_PERMUTATIONS; ++i) {
    slp_run(&slp, input_permutations[i]);
    printf("A: %.1f B: %.1f -> %.2f\n", input_permutations[i][0], input_permutations[i][1], slp.output);
  }
}

static void test_slp_or_gate(void) {
  // define inputs and expected outputs
  const float input_permutations[NUM_PERMUTATIONS][NUM_GATE_INPUTS] = {{0.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}, {1.0f, 0.0f}};
  const float expected_outputs[NUM_PERMUTATIONS] = {0.0f, 1.0f, 1.0f, 1.0f};
  float weights[NUM_GATE_INPUTS] = {0};

  const struct slp_sample samples[NUM_PERMUTATIONS] = {
      {.inputs = input_permutations[0],
       .output = expected_outputs[0]},
      {.inputs = input_permutations[1],
       .output = expected_outputs[1]},
      {.inputs = input_permutations[2],
       .output = expected_outputs[2]},
      {.inputs = input_permutations[3],
       .output = expected_outputs[3]},
  };
  uint16_t batch_sequence[NUM_PERMUTATIONS] = {0, 1, 2, 3};
  // train the slp
  struct slp_ctx slp = {
      .num_inputs = NUM_GATE_INPUTS,
      .activation = sigmoid_activation,
      .weights = weights,
  };
  struct slp_training_params params = {
      .epochs = 10000,
      .learning_rate = 0.01,
      .num_samples = NUM_PERMUTATIONS,
      .samples = samples,
      .sample_indices = batch_sequence,
  };
  slp_init(&slp);
  slp_train(&slp, &params);
  // printf("Training complete. Final Weights: %.2f %.2f Bias: %.2f\n", weights[0], weights[1], slp.bias);

  // run slp over each gate input permutation to verify results
  // printf(" --- Verification ---\n");
  for (uint8_t i = 0; i < NUM_PERMUTATIONS; ++i) {
    slp_run(&slp, input_permutations[i]);
    // printf("A: %.1f B: %.1f -> %.2f\n", input_permutations[i][0], input_permutations[i][1], slp.output);
  }
  fflush(stdout);
}

int main(int argc, char **argv) {
  // printf(" --- Single Layer Perceptron Verification ---\n");
  // printf(" --- Case 1 : AND Gate ---\n");
  // test_slp_and_gate();
  // printf(" --- Case 2 : OR Gate ---\n");
  test_slp_or_gate();
}
