#pragma once

#include "ctranslate2/generation.h"
#include "ctranslate2/layers/whisper.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/replica_pool.h"
#include "ctranslate2/vocabulary.h"

namespace ctranslate2 {
namespace models {

struct WhisperOptions {
size_t beam_size = 5;

float patience = 1;

float length_penalty = 1;

float repetition_penalty = 1;

size_t no_repeat_ngram_size = 0;

size_t max_length = 448;

size_t sampling_topk = 1;

float sampling_temperature = 1;

size_t num_hypotheses = 1;

bool return_scores = false;

bool return_no_speech_prob = false;

size_t max_initial_timestamp_index = 50;

bool suppress_blank = true;

std::vector<int> suppress_tokens = {-1};
};

struct WhisperGenerationResult {
std::vector<std::vector<std::string>> sequences;
std::vector<std::vector<size_t>> sequences_ids;
std::vector<float> scores;
float no_speech_prob = 0;

size_t num_sequences() const {
return sequences.size();
}

bool has_scores() const {
return !scores.empty();
}
};

struct WhisperAlignmentResult {
std::vector<std::pair<dim_t, dim_t>> alignments;
std::vector<float> text_token_probs;
};

class WhisperModel : public Model {
public:
const Vocabulary& get_vocabulary() const;

size_t current_spec_revision() const override;
bool is_quantizable(const std::string& variable_name) const override;
bool is_linear_weight(const std::string& variable_name) const override;
std::unique_ptr<Model> clone() const override;

bool use_global_int16_scale() const override {
return false;
}

protected:
void initialize(ModelReader& model_reader) override;

private:
std::shared_ptr<const Vocabulary> _vocabulary;
};

class WhisperReplica : public ModelReplica {
public:
static std::unique_ptr<WhisperReplica> create_from_model(const Model& model);

WhisperReplica(const std::shared_ptr<const WhisperModel>& model);

bool is_multilingual() const {
return _is_multilingual;
}

StorageView encode(StorageView features, const bool to_cpu);

std::vector<WhisperGenerationResult>
generate(StorageView features,
const std::vector<std::vector<std::string>>& prompts,
const WhisperOptions& options);

std::vector<WhisperGenerationResult>
generate(StorageView features,
const std::vector<std::vector<size_t>>& prompts,
const WhisperOptions& options);

std::vector<std::vector<std::pair<std::string, float>>>
detect_language(StorageView features);

std::vector<WhisperAlignmentResult>
align(StorageView features,
const std::vector<size_t>& start_sequence,
const std::vector<std::vector<size_t>>& text_tokens,
std::vector<size_t> num_frames,
dim_t median_filter_width);

private:
const std::shared_ptr<const WhisperModel> _model;
const std::unique_ptr<layers::WhisperEncoder> _encoder;
const std::unique_ptr<layers::WhisperDecoder> _decoder;

size_t _sot_id;
size_t _eot_id;
size_t _no_timestamps_id;
size_t _no_speech_id;
bool _is_multilingual;

StorageView maybe_encode(StorageView features);
};

class Whisper : public ReplicaPool<WhisperReplica> {
public:
using ReplicaPool::ReplicaPool;

bool is_multilingual() const;

std::future<StorageView> encode(const StorageView& features, const bool to_cpu);

std::vector<std::future<WhisperGenerationResult>>
generate(const StorageView& features,
std::vector<std::vector<std::string>> prompts,
WhisperOptions options = {});

std::vector<std::future<WhisperGenerationResult>>
generate(const StorageView& features,
std::vector<std::vector<size_t>> prompts,
WhisperOptions options = {});

std::vector<std::future<std::vector<std::pair<std::string, float>>>>
detect_language(const StorageView& features);

std::vector<std::future<WhisperAlignmentResult>>
align(const StorageView& features,
std::vector<size_t> start_sequence,
std::vector<std::vector<size_t>> text_tokens,
std::vector<size_t> num_frames,
dim_t median_filter_width);

};

}
}
