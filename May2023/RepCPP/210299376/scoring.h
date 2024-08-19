#pragma once

#include <numeric>
#include <string>
#include <vector>

#include "layers/decoder.h"
#include "vocabulary.h"

namespace ctranslate2 {

struct ScoringOptions {
size_t max_input_length = 1024;
};

struct ScoringResult {
std::vector<std::string> tokens;
std::vector<float> tokens_score;

float cumulated_score() const {
return std::accumulate(tokens_score.begin(), tokens_score.end(), 0.f);
}

float normalized_score() const {
const size_t num_tokens = tokens_score.size();
if (num_tokens == 0)
return 0.f;
return cumulated_score() / static_cast<float>(num_tokens);
}
};

std::vector<ScoringResult>
score_sequences(layers::Decoder& decoder,
layers::DecoderState& state,
const std::vector<std::vector<size_t>>& sequences,
const Vocabulary& vocabulary,
const dim_t preferred_size_multiple = 1);

}
