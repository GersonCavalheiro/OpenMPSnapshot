#pragma once

#include <variant>
#include <vector>
#include <string>

#include "decoding.h"
#include "vocabulary.h"

namespace ctranslate2 {

struct GenerationStepResult;

struct GenerationOptions {
size_t beam_size = 1;
float patience = 1;
float length_penalty = 1;
float repetition_penalty = 1;
size_t no_repeat_ngram_size = 0;
bool disable_unk = false;
std::vector<std::vector<std::string>> suppress_sequences;

std::variant<std::string, std::vector<std::string>, std::vector<size_t>> end_token;

bool return_end_token = false;

size_t max_length = 512;
size_t min_length = 0;

size_t sampling_topk = 1;
float sampling_temperature = 1;

size_t num_hypotheses = 1;

bool return_scores = false;

bool return_alternatives = false;
float min_alternative_expansion_prob = 0;

bool include_prompt_in_result = true;

std::function<void(GenerationStepResult)> callback = nullptr;
};

struct GenerationResult {
std::vector<std::vector<std::string>> sequences;
std::vector<std::vector<size_t>> sequences_ids;
std::vector<float> scores;

size_t num_sequences() const {
return sequences.size();
}

bool has_scores() const {
return !scores.empty();
}
};

struct GenerationStepResult {
size_t step;
size_t batch_id;
size_t token_id;
std::string token;
std::optional<float> log_prob;
bool is_last;

GenerationStepResult() = default;
GenerationStepResult(const DecodingStepResult& result, const Vocabulary& vocabulary)
: step(result.step)
, batch_id(result.batch_id)
, token_id(result.token_id)
, token(vocabulary.to_token(result.token_id))
, log_prob(result.log_prob)
, is_last(result.is_last)
{
}
};

class ResolveEndToken {
private:
const Vocabulary& _vocabulary;

public:
ResolveEndToken(const Vocabulary& vocabulary)
: _vocabulary(vocabulary)
{
}

std::vector<size_t> operator()(const std::string& token) const {
if (token.empty())
return {_vocabulary.eos_id()};
return {_vocabulary.to_id(token, false)};
}

std::vector<size_t> operator()(const std::vector<std::string>& tokens) const {
std::vector<size_t> ids;
ids.reserve(tokens.size());
for (const auto& token : tokens)
ids.emplace_back(_vocabulary.to_id(token, false));
return ids;
}

std::vector<size_t> operator()(const std::vector<size_t>& tokens) const {
return tokens;
}
};

}
