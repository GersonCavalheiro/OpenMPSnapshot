#pragma once

#include "translator.h"

namespace ctranslate2 {

class BufferedTranslationWrapper {
public:
BufferedTranslationWrapper(std::shared_ptr<Translator> translator,
size_t max_batch_size,
size_t buffer_timeout_in_micros,
TranslationOptions options = TranslationOptions(),
size_t max_buffer_size = 0);
~BufferedTranslationWrapper();

std::future<TranslationResult>
translate_async(std::vector<std::string> source, std::vector<std::string> target = {});

std::vector<std::future<TranslationResult>>
translate_batch_async(std::vector<std::vector<std::string>> source,
std::vector<std::vector<std::string>> target = {});

private:
std::shared_ptr<Translator> _translator;
const TranslationOptions _options;
const size_t _max_batch_size;
const size_t _max_buffer_size;
const std::chrono::microseconds _buffer_timeout;
std::unique_ptr<std::thread> _background_thread;
bool _stop = false;

std::mutex _mutex;
std::condition_variable _cv;
std::queue<Example> _examples;
std::queue<std::promise<TranslationResult>> _promises;

void buffer_loop();
};

}
