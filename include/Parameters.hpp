#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace wavo {

namespace detail {

template <typename T>
struct is_vector : std::false_type {};
template <typename U, typename A>
struct is_vector<std::vector<U, A>> : std::true_type {};

template <typename T, typename First, typename... Rest>
struct index_of : std::integral_constant<std::size_t, 1 + index_of<T, Rest...>::value> {};
template <typename T, typename... Rest>
struct index_of<T, T, Rest...> : std::integral_constant<std::size_t, 0> {};

template <typename T, typename V>
struct alternative_index;
template <typename T, typename... Ts>
struct alternative_index<T, std::variant<Ts...>> : index_of<T, Ts...> {};

template <typename T, typename V>
struct is_alternative : std::false_type {};
template <typename T, typename... Ts>
struct is_alternative<T, std::variant<Ts...>> : std::disjunction<std::is_same<T, Ts>...> {};

}  // namespace detail

/// Flat, heterogeneous key -> value store for pipeline/class configuration.
///
/// Values are one of the canonical config types in Value — everything a YAML
/// file or a Python dict can express. set() converts any arithmetic type,
/// string, or vector thereof into the canonical one; get<T>() converts back,
/// so a value written as `50` can be read as int, size_t, float, ... bool
/// never converts to/from numbers, and nothing converts to std::string.
///
/// Consumers should declare a ParameterSchema (see below) and validate their
/// scoped() slice on construction: that checks presence and types of every
/// key at once, fills defaults, and catches config typos as unknown keys.
///
/// fromYaml()/fromYamlString() parse with yaml-cpp (an implementation detail
/// of src/Parameters.cpp — this header stays STL-only) and flatten the
/// document into this store:
///  - nested maps become dotted keys ("pipeline.registration.alpha")
///  - scalars are typed as bool / std::int64_t / double / std::string;
///    quoted scalars are always strings; null / ~ values are skipped
///  - sequences of scalars load as one vector: all-int ->
///    std::vector<std::int64_t>, numeric -> std::vector<double>, all-bool ->
///    std::vector<bool>, otherwise std::vector<std::string>
///  - other sequences (of maps or nested sequences) are flattened with a
///    numeric segment per element: "stages.0.name", "stages.1.name", ...
///  - anchors/aliases and merge keys ("<<: *defaults") are resolved
class Parameters {
 public:
  /// Canonical config value types, in the order reported by type_name().
  using Value = std::variant<bool, std::int64_t, double, std::string, std::vector<std::int64_t>,
                             std::vector<double>, std::vector<bool>, std::vector<std::string>>;

  Parameters() = default;

  /// Loads a YAML file. Throws std::runtime_error on I/O or parse errors.
  static Parameters fromYaml(const std::string& path);
  /// Same, from an in-memory YAML string (handy for tests).
  static Parameters fromYamlString(const std::string& text);

  /// Canonicalizes a supported type: bool stays bool, other integrals ->
  /// std::int64_t, floating point -> double, string-likes -> std::string,
  /// vectors elementwise the same way. Unsupported types fail to compile.
  template <typename T>
  static Value toValue(T value) {
    using D = std::decay_t<T>;
    if constexpr (detail::is_vector<D>::value) {
      using U = typename D::value_type;
      if constexpr (std::is_same_v<U, bool>) {
        return value;
      } else if constexpr (std::is_integral_v<U>) {
        std::vector<std::int64_t> out;
        out.reserve(value.size());
        for (const auto& e : value) out.push_back(static_cast<std::int64_t>(e));
        return out;
      } else if constexpr (std::is_floating_point_v<U>) {
        std::vector<double> out(value.begin(), value.end());
        return out;
      } else {
        return std::vector<std::string>(value.begin(), value.end());
      }
    } else if constexpr (std::is_same_v<D, bool>) {
      return value;
    } else if constexpr (std::is_integral_v<D>) {
      return static_cast<std::int64_t>(value);
    } else if constexpr (std::is_floating_point_v<D>) {
      return static_cast<double>(value);
    } else {
      return Value(std::move(value));  // std::string, const char*, Value
    }
  }

  /// Index of the canonical alternative T maps to (see toValue()).
  template <typename T>
  static constexpr std::size_t canonicalIndex() {
    using D = std::decay_t<T>;
    if constexpr (detail::is_vector<D>::value) {
      using U = typename D::value_type;
      if constexpr (std::is_same_v<U, bool>)
        return detail::alternative_index<std::vector<bool>, Value>::value;
      else if constexpr (std::is_integral_v<U>)
        return detail::alternative_index<std::vector<std::int64_t>, Value>::value;
      else if constexpr (std::is_floating_point_v<U>)
        return detail::alternative_index<std::vector<double>, Value>::value;
      else
        return detail::alternative_index<std::vector<std::string>, Value>::value;
    } else if constexpr (std::is_same_v<D, bool>) {
      return detail::alternative_index<bool, Value>::value;
    } else if constexpr (std::is_integral_v<D>) {
      return detail::alternative_index<std::int64_t, Value>::value;
    } else if constexpr (std::is_floating_point_v<D>) {
      return detail::alternative_index<double, Value>::value;
    } else {
      return detail::alternative_index<std::string, Value>::value;
    }
  }

  /// Human-readable name of a Value alternative, by index().
  static const char* type_name(std::size_t index) {
    static constexpr const char* kNames[] = {"bool",      "int",         "float",      "str",
                                             "list[int]", "list[float]", "list[bool]", "list[str]"};
    static_assert(sizeof(kNames) / sizeof(kNames[0]) == std::variant_size_v<Value>);
    return kNames[index];
  }

  template <typename T>
  void set(const std::string& key, T value) {
    values_[key] = toValue(std::move(value));
  }

  bool has(const std::string& key) const { return values_.count(key) != 0; }

  /// Throws std::out_of_range on a missing key, std::runtime_error when the
  /// stored value is not convertible to T.
  template <typename T>
  T get(const std::string& key) const {
    return cast<T>(raw(key), key);
  }

  /// Missing key returns `fallback`; a present key of the wrong type still
  /// throws (a typo'd type in the config should not be silently ignored).
  template <typename T>
  T get(const std::string& key, T fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    return cast<T>(it->second, key);
  }

  /// Copy of the entries under "<prefix>." with the prefix stripped, so a
  /// pipeline can hand each component its own sub-config.
  Parameters scoped(const std::string& prefix) const {
    Parameters out;
    const std::string p = prefix + ".";
    for (const auto& [key, value] : values_)
      if (key.rfind(p, 0) == 0) out.values_[key.substr(p.size())] = value;
    return out;
  }

  /// Type-erased access to a stored value (for bindings/debugging). Throws
  /// std::out_of_range on a missing key.
  const Value& raw(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) throw std::out_of_range("Parameters: missing key '" + key + "'");
    return it->second;
  }

  std::vector<std::string> keys() const {
    std::vector<std::string> out;
    out.reserve(values_.size());
    for (const auto& [key, value] : values_) out.push_back(key);
    return out;
  }

 private:
  template <typename T>
  static T cast(const Value& v, const std::string& key) {
    if constexpr (detail::is_alternative<T, Value>::value) {
      if (const T* p = std::get_if<T>(&v)) return *p;
    }
    if constexpr (std::is_arithmetic_v<T> && !std::is_same_v<T, bool>) {
      if (const auto* p = std::get_if<std::int64_t>(&v)) return static_cast<T>(*p);
      if (const auto* p = std::get_if<double>(&v)) return static_cast<T>(*p);
    } else if constexpr (detail::is_vector<T>::value) {
      using U = typename T::value_type;
      if constexpr (std::is_arithmetic_v<U> && !std::is_same_v<U, bool>) {
        if (const auto* p = std::get_if<std::vector<std::int64_t>>(&v)) {
          T out;
          out.reserve(p->size());
          for (const auto e : *p) out.push_back(static_cast<U>(e));
          return out;
        }
        if (const auto* p = std::get_if<std::vector<double>>(&v)) {
          T out;
          out.reserve(p->size());
          for (const auto e : *p) out.push_back(static_cast<U>(e));
          return out;
        }
      }
      // An empty YAML/Python list carries no element type; hand it out as
      // any vector.
      if (const auto* p = std::get_if<std::vector<std::string>>(&v); p && p->empty()) return T{};
    }
    throw std::runtime_error("Parameters: key '" + key + "' holds " + type_name(v.index()) +
                             ", requested an incompatible type");
  }

  std::map<std::string, Value> values_;
};

/// Declares the parameters a consumer expects: key, canonical type, and an
/// optional default. Consumers typically expose a static schema and validate
/// their config slice on construction:
///
///   static const ParameterSchema& schema() {
///     static const auto s = ParameterSchema("registration")
///         .require<double>("tolerance")
///         .optional<int>("iterations", 50);
///     return s;
///   }
///   explicit ImageRegistrator(const Parameters& p) : cfg_(schema().validate(p)) {}
class ParameterSchema {
 public:
  explicit ParameterSchema(std::string name = "") : name_(std::move(name)) {}

  template <typename T>
  ParameterSchema& require(const std::string& key) {
    specs_[key] = Spec{Parameters::canonicalIndex<T>(), std::nullopt};
    return *this;
  }

  template <typename T>
  ParameterSchema& optional(const std::string& key, T default_value) {
    Parameters::Value v = Parameters::toValue(std::move(default_value));
    const std::size_t index = v.index();
    specs_[key] = Spec{index, std::move(v)};
    return *this;
  }

  /// Returns `params` coerced to the declared types with defaults filled in.
  /// Throws one std::runtime_error listing every problem at once: missing
  /// required keys, type mismatches, and — unless allow_unknown — keys not
  /// declared in the schema (i.e. config typos).
  Parameters validate(const Parameters& params, bool allow_unknown = false) const {
    std::vector<std::string> errors;
    Parameters out;
    for (const auto& [key, spec] : specs_) {
      if (!params.has(key)) {
        if (spec.default_value)
          out.set(key, *spec.default_value);
        else
          errors.push_back("missing required key '" + key + "' (" +
                           Parameters::type_name(spec.type_index) + ")");
        continue;
      }
      Parameters::Value coerced;
      if (coerce(params.raw(key), spec.type_index, &coerced))
        out.set(key, std::move(coerced));
      else
        errors.push_back("key '" + key + "': expected " + Parameters::type_name(spec.type_index) +
                         ", got " + Parameters::type_name(params.raw(key).index()));
    }
    if (!allow_unknown)
      for (const auto& key : params.keys())
        if (specs_.count(key) == 0) errors.push_back("unknown key '" + key + "'");
    if (!errors.empty()) {
      std::string msg =
          (name_.empty() ? std::string("Parameters") : name_) + ": invalid configuration:";
      for (const auto& e : errors) msg += "\n  - " + e;
      throw std::runtime_error(msg);
    }
    return out;
  }

 private:
  struct Spec {
    std::size_t type_index = 0;
    std::optional<Parameters::Value> default_value;
  };

  /// Same-type always coerces; int widens to float (scalar and list); an
  /// empty list adapts to any list type. Everything else is a mismatch.
  static bool coerce(const Parameters::Value& in, std::size_t target, Parameters::Value* out) {
    if (in.index() == target) {
      *out = in;
      return true;
    }
    if (target == Parameters::canonicalIndex<double>()) {
      if (const auto* p = std::get_if<std::int64_t>(&in)) {
        *out = static_cast<double>(*p);
        return true;
      }
    }
    if (target == Parameters::canonicalIndex<std::vector<double>>()) {
      if (const auto* p = std::get_if<std::vector<std::int64_t>>(&in)) {
        *out = std::vector<double>(p->begin(), p->end());
        return true;
      }
    }
    if (const auto* p = std::get_if<std::vector<std::string>>(&in); p && p->empty()) {
      if (target == Parameters::canonicalIndex<std::vector<std::int64_t>>())
        *out = std::vector<std::int64_t>{};
      else if (target == Parameters::canonicalIndex<std::vector<double>>())
        *out = std::vector<double>{};
      else if (target == Parameters::canonicalIndex<std::vector<bool>>())
        *out = std::vector<bool>{};
      else
        return false;
      return true;
    }
    return false;
  }

  std::string name_;
  std::map<std::string, Spec> specs_;
};

}  // namespace wavo
