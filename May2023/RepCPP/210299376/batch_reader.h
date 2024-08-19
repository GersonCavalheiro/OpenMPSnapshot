#pragma once

#include <istream>
#include <memory>
#include <string>
#include <vector>

#include "utils.h"

namespace ctranslate2 {

enum class BatchType {
Examples,
Tokens,
};

BatchType str_to_batch_type(const std::string& batch_type);


struct Example {
std::vector<std::vector<std::string>> streams;

Example() = default;

Example(std::vector<std::string> sequence) {
streams.emplace_back(std::move(sequence));
}

Example(std::vector<std::string> source, std::vector<std::string> target) {
streams.reserve(2);
streams.emplace_back(std::move(source));
streams.emplace_back(std::move(target));
}

size_t num_streams() const {
return streams.size();
}

bool empty() const {
return streams.empty();
}

size_t length(size_t index = 0) const {
if (index >= streams.size())
return 0;
return streams[index].size();
}
};


class BatchReader {
public:
virtual ~BatchReader() = default;

std::vector<Example>
get_next(const size_t max_batch_size,
const BatchType batch_type = BatchType::Examples);

virtual Example get_next_example() = 0;

virtual size_t num_examples() const {
return 0;
}

private:
bool _initialized = false;
Example _next;
};

template <typename Tokenizer>
class TextLineReader : public BatchReader {
public:
TextLineReader(std::istream& stream, Tokenizer& tokenizer)
: _stream(stream)
, _tokenizer(tokenizer)
{
}

Example get_next_example() override {
Example example;

std::string line;
if (ctranslate2::getline(_stream, line))
example.streams.emplace_back(_tokenizer(line));

return example;
}

private:
std::istream& _stream;
Tokenizer& _tokenizer;
};

class VectorReader : public BatchReader {
public:
VectorReader(std::vector<std::vector<std::string>> examples);
VectorReader(std::vector<Example> examples);

Example get_next_example() override;
size_t num_examples() const override {
return _examples.size();
}

private:
std::vector<Example> _examples;
size_t _index = 0;
};

class ParallelBatchReader : public BatchReader {
public:
void add(std::unique_ptr<BatchReader> reader);

Example get_next_example() override;
size_t num_examples() const override;

private:
std::vector<std::unique_ptr<BatchReader>> _readers;
};


struct Batch {
std::vector<Example> examples;
std::vector<size_t> example_index;  

size_t num_examples() const {
return examples.size();
}

bool empty() const {
return examples.empty();
}

std::vector<std::vector<std::string>> get_stream(size_t index) const;
};


std::vector<Example>
load_examples(std::vector<std::vector<std::vector<std::string>>> streams);

std::vector<Batch>
rebatch_input(const std::vector<Example>& examples,
size_t max_batch_size = 0,
BatchType batch_type = BatchType::Examples);

}
