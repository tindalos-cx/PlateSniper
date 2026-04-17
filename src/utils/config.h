#pragma once

#include <string>
#include <map>

namespace platesniper {

class Config {
public:
    static Config& instance();

    bool load(const std::string& config_path);
    bool save(const std::string& config_path);

    std::string getString(const std::string& key, const std::string& default_value = "") const;
    int getInt(const std::string& key, int default_value = 0) const;
    float getFloat(const std::string& key, float default_value = 0.0f) const;
    bool getBool(const std::string& key, bool default_value = false) const;

    void setString(const std::string& key, const std::string& value);
    void setInt(const std::string& key, int value);
    void setFloat(const std::string& key, float value);
    void setBool(const std::string& key, bool value);

    void reset();

private:
    Config() = default;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    std::map<std::string, std::string> values_;

    std::string trim(const std::string& str) const;
};

} // namespace platesniper
