#pragma once

#include <istream>
#include <memory>
#include <string>
#include <unordered_map>

namespace ctranslate2 {
namespace models {

class ModelReader {
public:
virtual ~ModelReader() = default;

virtual std::string get_model_id() const = 0;
virtual std::unique_ptr<std::istream> get_file(const std::string& filename,
const bool binary = false) = 0;

std::unique_ptr<std::istream> get_required_file(const std::string& filename,
const bool binary = false);
};

class ModelFileReader : public ModelReader {
public:
ModelFileReader(std::string model_dir);
std::string get_model_id() const override;
std::unique_ptr<std::istream> get_file(const std::string& filename,
const bool binary = false) override;

private:
std::string _model_dir;
};

class ModelMemoryReader : public ModelReader {
public:
ModelMemoryReader(std::string model_name);

void register_file(std::string filename, std::string content);

std::string get_model_id() const override;
std::unique_ptr<std::istream> get_file(const std::string& filename,
const bool binary = false) override;

private:
std::string _model_name;
std::unordered_map<std::string, std::string> _files;
};

}
}
