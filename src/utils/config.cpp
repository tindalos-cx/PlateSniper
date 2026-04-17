#include "utils/config.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace platesniper {

Config& Config::instance() {
    static Config instance;
    return instance;
}

bool Config::load(const std::string& config_path) {
    std::ifstream file(config_path);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        line = trim(line);

        if (line.empty() || line[0] == '#' || line[0] == ';') {
            continue;
        }

        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = trim(line.substr(0, pos));
            std::string value = trim(line.substr(pos + 1));

            if (!key.empty()) {
                values_[key] = value;
            }
        }
    }

    return true;
}

bool Config::save(const std::string& config_path) {
    std::ofstream file(config_path);
    if (!file.is_open()) {
        return false;
    }

    for (const auto& pair : values_) {
        file << pair.first << " = " << pair.second << "\n";
    }

    return true;
}

std::string Config::getString(const std::string& key, const std::string& default_value) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        return it->second;
    }
    return default_value;
}

int Config::getInt(const std::string& key, int default_value) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        try {
            return std::stoi(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

float Config::getFloat(const std::string& key, float default_value) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        try {
            return std::stof(it->second);
        } catch (...) {
            return default_value;
        }
    }
    return default_value;
}

bool Config::getBool(const std::string& key, bool default_value) const {
    auto it = values_.find(key);
    if (it != values_.end()) {
        std::string value = it->second;
        std::transform(value.begin(), value.end(), value.begin(), ::tolower);
        if (value == "true" || value == "1" || value == "yes" || value == "on") {
            return true;
        }
        if (value == "false" || value == "0" || value == "no" || value == "off") {
            return false;
        }
    }
    return default_value;
}

void Config::setString(const std::string& key, const std::string& value) {
    values_[key] = value;
}

void Config::setInt(const std::string& key, int value) {
    values_[key] = std::to_string(value);
}

void Config::setFloat(const std::string& key, float value) {
    values_[key] = std::to_string(value);
}

void Config::setBool(const std::string& key, bool value) {
    values_[key] = value ? "true" : "false";
}

void Config::reset() {
    values_.clear();
}

std::string Config::trim(const std::string& str) const {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return "";
    }
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

} // namespace platesniper
