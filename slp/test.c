
#include "slp.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define NUM_PERMUTATIONS 4
#define NUM_GATE_INPUTS 2
#define TRAIN_EPOCHS (uint16_t)1000

float step(const float in) {
  return in >= 0 ? 1.0f : 0.0f;
}

static void test_slp_and_gate(void) {
  // define inputs and expected outputs
  const float input_permutations[NUM_PERMUTATIONS][NUM_GATE_INPUTS] = {{0, 0}, {0, 1}, {1, 1}, {1, 0}};
  const float expected_outputs[NUM_PERMUTATIONS] = {0, 0, 1, 0};
  float weights[NUM_GATE_INPUTS] = {0};

  // train the slp
  struct slp_ctx slp = {
      .num_inputs = NUM_GATE_INPUTS,
      .activation = step,
      .weights = weights,
  };
  struct slp_training_params params = {
      .epochs = 1000,
      .learning_rate = 0.1,
      .num_samples = NUM_PERMUTATIONS,
      .inputs = input_permutations,
      .outputs = expected_outputs,
  };
  slp_train(&slp, &params);
  printf("Training complete. Final Weights: %.2f %.2f Bias: %.2f\n", weights[0], weights[1], slp.bias);

  // run slp over each gate input permutation to verify results
  for (uint8_t i = 0; i < NUM_PERMUTATIONS; ++i) {
    slp_run(&slp, input_permutations[i]);
    printf("A: %.1f B: %.1f -> %.2f\n", input_permutations[i][0], input_permutations[i][1], slp.output);
  }
}

int main(int argc, char **argv) {
  printf(" --- Single Layer Perceptron Verification ---\n");
  printf(" --- Case 1 : AND Gate ---\n");
  test_slp_and_gate();
}
