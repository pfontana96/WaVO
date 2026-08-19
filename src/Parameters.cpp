#include "Parameters.hpp"

#include <fstream>
#include <set>
#include <sstream>

#include <yaml-cpp/yaml.h>

namespace wavo {
namespace {

/// Scalar -> bool / int64 / double, falling back to the raw text. A quoted
/// scalar (tag "!") is always a string, so "50" stays "50".
Parameters::Value ScalarToValue(const YAML::Node& node) {
  if (node.Tag() != "!") {
    bool b;
    if (YAML::convert<bool>::decode(node, b)) return b;
    std::int64_t i;
    if (YAML::convert<std::int64_t>::decode(node, i)) return i;
    double d;
    if (YAML::convert<double>::decode(node, d)) return d;
  }
  return node.Scalar();
}

/// Homogenizes a scalar-only sequence: all ints -> vector<int64_t>, numeric ->
/// vector<double>, all bools -> vector<bool>, otherwise vector<string> (mixed
/// lists fall back to each element's raw text).
Parameters::Value MakeVector(const YAML::Node& seq) {
  std::vector<Parameters::Value> parsed;
  parsed.reserve(seq.size());
  for (const auto& item : seq) parsed.push_back(ScalarToValue(item));
  if (parsed.empty()) return std::vector<std::string>{};
  bool all_int = true, all_num = true, all_bool = true;
  for (const auto& v : parsed) {
    const bool is_int = std::holds_alternative<std::int64_t>(v);
    all_int &= is_int;
    all_num &= is_int || std::holds_alternative<double>(v);
    all_bool &= std::holds_alternative<bool>(v);
  }
  if (all_int) {
    std::vector<std::int64_t> out;
    for (const auto& v : parsed) out.push_back(std::get<std::int64_t>(v));
    return out;
  }
  if (all_num) {
    std::vector<double> out;
    for (const auto& v : parsed) {
      if (const auto* p = std::get_if<std::int64_t>(&v))
        out.push_back(static_cast<double>(*p));
      else
        out.push_back(std::get<double>(v));
    }
    return out;
  }
  if (all_bool) {
    std::vector<bool> out;
    for (const auto& v : parsed) out.push_back(std::get<bool>(v));
    return out;
  }
  std::vector<std::string> out;
  for (const auto& item : seq) out.push_back(item.Scalar());
  return out;
}

/// Resolves YAML 1.1 merge keys ("<<: *anchor"), which yaml-cpp keeps as
/// literal entries. Explicit keys win over merged ones; with a sequence of
/// merge maps, earlier maps win over later ones (per the spec). A quoted
/// "<<" stays a plain key.
void CollectMapEntries(const YAML::Node& map, std::set<std::string>* seen,
                       std::vector<std::pair<std::string, YAML::Node>>* entries) {
  std::vector<YAML::Node> merges;
  for (const auto& kv : map) {
    const std::string key = kv.first.as<std::string>();
    if (key == "<<" && kv.first.Tag() != "!") {
      if (kv.second.IsSequence())
        for (const auto& m : kv.second) merges.push_back(m);
      else
        merges.push_back(kv.second);
    } else if (seen->insert(key).second) {
      entries->emplace_back(key, kv.second);
    }
  }
  for (const auto& m : merges)
    if (m.IsMap()) CollectMapEntries(m, seen, entries);
}

void Flatten(const YAML::Node& node, const std::string& prefix,
             std::map<std::string, Parameters::Value>* out) {
  switch (node.Type()) {
    case YAML::NodeType::Scalar:
      (*out)[prefix] = ScalarToValue(node);
      break;
    case YAML::NodeType::Sequence: {
      bool all_scalar = true;
      for (const auto& item : node)
        if (!item.IsScalar()) {
          all_scalar = false;
          break;
        }
      if (all_scalar) {
        (*out)[prefix] = MakeVector(node);
      } else {
        // Sequence of maps/sequences: one numeric segment per element.
        std::size_t i = 0;
        for (const auto& item : node) Flatten(item, prefix + "." + std::to_string(i++), out);
      }
      break;
    }
    case YAML::NodeType::Map: {
      std::set<std::string> seen;
      std::vector<std::pair<std::string, YAML::Node>> entries;
      CollectMapEntries(node, &seen, &entries);
      for (const auto& [key, child] : entries)
        Flatten(child, prefix.empty() ? key : prefix + "." + key, out);
      break;
    }
    case YAML::NodeType::Null:
    case YAML::NodeType::Undefined:
      break;  // null values leave the key absent, so get(key, fallback) kicks in
  }
}

}  // namespace

Parameters Parameters::fromYamlString(const std::string& text) {
  YAML::Node root;
  try {
    root = YAML::Load(text);
  } catch (const YAML::Exception& e) {
    throw std::runtime_error(std::string("Parameters: YAML parse error: ") + e.what());
  }
  Parameters out;
  if (root.IsNull() || !root.IsDefined()) return out;
  if (!root.IsMap()) throw std::runtime_error("Parameters: top-level YAML node must be a map");
  std::map<std::string, Parameters::Value> values;
  Flatten(root, "", &values);
  for (auto& [key, value] : values) out.set(key, std::move(value));
  return out;
}

Parameters Parameters::fromYaml(const std::string& path) {
  std::ifstream file(path);
  if (!file) throw std::runtime_error("Parameters: cannot open '" + path + "'");
  std::ostringstream buffer;
  buffer << file.rdbuf();
  return fromYamlString(buffer.str());
}

}  // namespace wavo
