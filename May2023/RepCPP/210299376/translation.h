#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include "generation.h"

namespace ctranslate2 {

struct TranslationOptions {
size_t beam_size = 2;
float patience = 1;
float length_penalty = 1;
float coverage_penalty = 0;
float repetition_penalty = 1;
size_t no_repeat_ngram_size = 0;
bool disable_unk = false;
std::vector<std::vector<std::string>> suppress_sequences;
float prefix_bias_beta = 0;

std::variant<std::string, std::vector<std::string>, std::vector<size_t>> end_token;

bool return_end_token = false;

size_t max_input_length = 1024;

size_t max_decoding_length = 256;
size_t min_decoding_length = 1;

size_t sampling_topk = 1;
float sampling_temperature = 1;

bool use_vmap = false;

size_t num_hypotheses = 1;

bool return_scores = false;
bool return_attention = false;

bool return_alternatives = false;
float min_alternative_expansion_prob = 0;

bool replace_unknowns = false;

std::function<void(GenerationStepResult)> callback = nullptr;
};

struct TranslationResult {
std::vector<std::vector<std::string>> hypotheses;
std::vector<float> scores;
std::vector<std::vector<std::vector<float>>> attention;

TranslationResult(std::vector<std::vector<std::string>> hypotheses_)
: hypotheses(std::move(hypotheses_))
{
}

TranslationResult(std::vector<std::vector<std::string>> hypotheses_,
std::vector<float> scores_,
std::vector<std::vector<std::vector<float>>> attention_)
: hypotheses(std::move(hypotheses_))
, scores(std::move(scores_))
, attention(std::move(attention_))
{
}

TranslationResult(const size_t num_hypotheses,
const bool with_attention,
const bool with_score)
: hypotheses(num_hypotheses)
, scores(with_score ? num_hypotheses : 0, static_cast<float>(0))
, attention(with_attention ? num_hypotheses : 0)
{
}

TranslationResult() = default;

const std::vector<std::string>& output() const {
if (hypotheses.empty())
throw std::runtime_error("This result is empty");
return hypotheses[0];
}

float score() const {
if (scores.empty())
throw std::runtime_error("This result has no scores");
return scores[0];
}

size_t num_hypotheses() const {
return hypotheses.size();
}

bool has_scores() const {
return !scores.empty();
}

bool has_attention() const {
return !attention.empty();
}
};

}
