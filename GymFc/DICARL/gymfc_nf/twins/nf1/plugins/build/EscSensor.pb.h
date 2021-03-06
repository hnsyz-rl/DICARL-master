// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: EscSensor.proto

#ifndef PROTOBUF_EscSensor_2eproto__INCLUDED
#define PROTOBUF_EscSensor_2eproto__INCLUDED

#include <string>

#include <google/protobuf/stubs/common.h>

#if GOOGLE_PROTOBUF_VERSION < 3000000
#error This file was generated by a newer version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please update
#error your headers.
#endif
#if 3000000 < GOOGLE_PROTOBUF_MIN_PROTOC_VERSION
#error This file was generated by an older version of protoc which is
#error incompatible with your Protocol Buffer headers.  Please
#error regenerate this file with a newer version of protoc.
#endif

#include <google/protobuf/arena.h>
#include <google/protobuf/arenastring.h>
#include <google/protobuf/generated_message_util.h>
#include <google/protobuf/metadata.h>
#include <google/protobuf/message.h>
#include <google/protobuf/repeated_field.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/unknown_field_set.h>
// @@protoc_insertion_point(includes)

namespace sensor_msgs {
namespace msgs {

// Internal implementation detail -- do not call these.
void protobuf_AddDesc_EscSensor_2eproto();
void protobuf_AssignDesc_EscSensor_2eproto();
void protobuf_ShutdownFile_EscSensor_2eproto();

class EscSensor;

// ===================================================================

class EscSensor : public ::google::protobuf::Message /* @@protoc_insertion_point(class_definition:sensor_msgs.msgs.EscSensor) */ {
 public:
  EscSensor();
  virtual ~EscSensor();

  EscSensor(const EscSensor& from);

  inline EscSensor& operator=(const EscSensor& from) {
    CopyFrom(from);
    return *this;
  }

  inline const ::google::protobuf::UnknownFieldSet& unknown_fields() const {
    return _internal_metadata_.unknown_fields();
  }

  inline ::google::protobuf::UnknownFieldSet* mutable_unknown_fields() {
    return _internal_metadata_.mutable_unknown_fields();
  }

  static const ::google::protobuf::Descriptor* descriptor();
  static const EscSensor& default_instance();

  void Swap(EscSensor* other);

  // implements Message ----------------------------------------------

  inline EscSensor* New() const { return New(NULL); }

  EscSensor* New(::google::protobuf::Arena* arena) const;
  void CopyFrom(const ::google::protobuf::Message& from);
  void MergeFrom(const ::google::protobuf::Message& from);
  void CopyFrom(const EscSensor& from);
  void MergeFrom(const EscSensor& from);
  void Clear();
  bool IsInitialized() const;

  int ByteSize() const;
  bool MergePartialFromCodedStream(
      ::google::protobuf::io::CodedInputStream* input);
  void SerializeWithCachedSizes(
      ::google::protobuf::io::CodedOutputStream* output) const;
  ::google::protobuf::uint8* InternalSerializeWithCachedSizesToArray(
      bool deterministic, ::google::protobuf::uint8* output) const;
  ::google::protobuf::uint8* SerializeWithCachedSizesToArray(::google::protobuf::uint8* output) const {
    return InternalSerializeWithCachedSizesToArray(false, output);
  }
  int GetCachedSize() const { return _cached_size_; }
  private:
  void SharedCtor();
  void SharedDtor();
  void SetCachedSize(int size) const;
  void InternalSwap(EscSensor* other);
  private:
  inline ::google::protobuf::Arena* GetArenaNoVirtual() const {
    return _internal_metadata_.arena();
  }
  inline void* MaybeArenaPtr() const {
    return _internal_metadata_.raw_arena_ptr();
  }
  public:

  ::google::protobuf::Metadata GetMetadata() const;

  // nested types ----------------------------------------------------

  // accessors -------------------------------------------------------

  // required uint32 id = 1;
  bool has_id() const;
  void clear_id();
  static const int kIdFieldNumber = 1;
  ::google::protobuf::uint32 id() const;
  void set_id(::google::protobuf::uint32 value);

  // required float motor_speed = 2;
  bool has_motor_speed() const;
  void clear_motor_speed();
  static const int kMotorSpeedFieldNumber = 2;
  float motor_speed() const;
  void set_motor_speed(float value);

  // required float temperature = 3;
  bool has_temperature() const;
  void clear_temperature();
  static const int kTemperatureFieldNumber = 3;
  float temperature() const;
  void set_temperature(float value);

  // required float voltage = 4;
  bool has_voltage() const;
  void clear_voltage();
  static const int kVoltageFieldNumber = 4;
  float voltage() const;
  void set_voltage(float value);

  // required float current = 5;
  bool has_current() const;
  void clear_current();
  static const int kCurrentFieldNumber = 5;
  float current() const;
  void set_current(float value);

  // required float force = 6;
  bool has_force() const;
  void clear_force();
  static const int kForceFieldNumber = 6;
  float force() const;
  void set_force(float value);

  // required float torque = 7;
  bool has_torque() const;
  void clear_torque();
  static const int kTorqueFieldNumber = 7;
  float torque() const;
  void set_torque(float value);

  // @@protoc_insertion_point(class_scope:sensor_msgs.msgs.EscSensor)
 private:
  inline void set_has_id();
  inline void clear_has_id();
  inline void set_has_motor_speed();
  inline void clear_has_motor_speed();
  inline void set_has_temperature();
  inline void clear_has_temperature();
  inline void set_has_voltage();
  inline void clear_has_voltage();
  inline void set_has_current();
  inline void clear_has_current();
  inline void set_has_force();
  inline void clear_has_force();
  inline void set_has_torque();
  inline void clear_has_torque();

  // helper for ByteSize()
  int RequiredFieldsByteSizeFallback() const;

  ::google::protobuf::internal::InternalMetadataWithArena _internal_metadata_;
  ::google::protobuf::uint32 _has_bits_[1];
  mutable int _cached_size_;
  ::google::protobuf::uint32 id_;
  float motor_speed_;
  float temperature_;
  float voltage_;
  float current_;
  float force_;
  float torque_;
  friend void  protobuf_AddDesc_EscSensor_2eproto();
  friend void protobuf_AssignDesc_EscSensor_2eproto();
  friend void protobuf_ShutdownFile_EscSensor_2eproto();

  void InitAsDefaultInstance();
  static EscSensor* default_instance_;
};
// ===================================================================


// ===================================================================

#if !PROTOBUF_INLINE_NOT_IN_HEADERS
// EscSensor

// required uint32 id = 1;
inline bool EscSensor::has_id() const {
  return (_has_bits_[0] & 0x00000001u) != 0;
}
inline void EscSensor::set_has_id() {
  _has_bits_[0] |= 0x00000001u;
}
inline void EscSensor::clear_has_id() {
  _has_bits_[0] &= ~0x00000001u;
}
inline void EscSensor::clear_id() {
  id_ = 0u;
  clear_has_id();
}
inline ::google::protobuf::uint32 EscSensor::id() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.id)
  return id_;
}
inline void EscSensor::set_id(::google::protobuf::uint32 value) {
  set_has_id();
  id_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.id)
}

// required float motor_speed = 2;
inline bool EscSensor::has_motor_speed() const {
  return (_has_bits_[0] & 0x00000002u) != 0;
}
inline void EscSensor::set_has_motor_speed() {
  _has_bits_[0] |= 0x00000002u;
}
inline void EscSensor::clear_has_motor_speed() {
  _has_bits_[0] &= ~0x00000002u;
}
inline void EscSensor::clear_motor_speed() {
  motor_speed_ = 0;
  clear_has_motor_speed();
}
inline float EscSensor::motor_speed() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.motor_speed)
  return motor_speed_;
}
inline void EscSensor::set_motor_speed(float value) {
  set_has_motor_speed();
  motor_speed_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.motor_speed)
}

// required float temperature = 3;
inline bool EscSensor::has_temperature() const {
  return (_has_bits_[0] & 0x00000004u) != 0;
}
inline void EscSensor::set_has_temperature() {
  _has_bits_[0] |= 0x00000004u;
}
inline void EscSensor::clear_has_temperature() {
  _has_bits_[0] &= ~0x00000004u;
}
inline void EscSensor::clear_temperature() {
  temperature_ = 0;
  clear_has_temperature();
}
inline float EscSensor::temperature() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.temperature)
  return temperature_;
}
inline void EscSensor::set_temperature(float value) {
  set_has_temperature();
  temperature_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.temperature)
}

// required float voltage = 4;
inline bool EscSensor::has_voltage() const {
  return (_has_bits_[0] & 0x00000008u) != 0;
}
inline void EscSensor::set_has_voltage() {
  _has_bits_[0] |= 0x00000008u;
}
inline void EscSensor::clear_has_voltage() {
  _has_bits_[0] &= ~0x00000008u;
}
inline void EscSensor::clear_voltage() {
  voltage_ = 0;
  clear_has_voltage();
}
inline float EscSensor::voltage() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.voltage)
  return voltage_;
}
inline void EscSensor::set_voltage(float value) {
  set_has_voltage();
  voltage_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.voltage)
}

// required float current = 5;
inline bool EscSensor::has_current() const {
  return (_has_bits_[0] & 0x00000010u) != 0;
}
inline void EscSensor::set_has_current() {
  _has_bits_[0] |= 0x00000010u;
}
inline void EscSensor::clear_has_current() {
  _has_bits_[0] &= ~0x00000010u;
}
inline void EscSensor::clear_current() {
  current_ = 0;
  clear_has_current();
}
inline float EscSensor::current() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.current)
  return current_;
}
inline void EscSensor::set_current(float value) {
  set_has_current();
  current_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.current)
}

// required float force = 6;
inline bool EscSensor::has_force() const {
  return (_has_bits_[0] & 0x00000020u) != 0;
}
inline void EscSensor::set_has_force() {
  _has_bits_[0] |= 0x00000020u;
}
inline void EscSensor::clear_has_force() {
  _has_bits_[0] &= ~0x00000020u;
}
inline void EscSensor::clear_force() {
  force_ = 0;
  clear_has_force();
}
inline float EscSensor::force() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.force)
  return force_;
}
inline void EscSensor::set_force(float value) {
  set_has_force();
  force_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.force)
}

// required float torque = 7;
inline bool EscSensor::has_torque() const {
  return (_has_bits_[0] & 0x00000040u) != 0;
}
inline void EscSensor::set_has_torque() {
  _has_bits_[0] |= 0x00000040u;
}
inline void EscSensor::clear_has_torque() {
  _has_bits_[0] &= ~0x00000040u;
}
inline void EscSensor::clear_torque() {
  torque_ = 0;
  clear_has_torque();
}
inline float EscSensor::torque() const {
  // @@protoc_insertion_point(field_get:sensor_msgs.msgs.EscSensor.torque)
  return torque_;
}
inline void EscSensor::set_torque(float value) {
  set_has_torque();
  torque_ = value;
  // @@protoc_insertion_point(field_set:sensor_msgs.msgs.EscSensor.torque)
}

#endif  // !PROTOBUF_INLINE_NOT_IN_HEADERS

// @@protoc_insertion_point(namespace_scope)

}  // namespace msgs
}  // namespace sensor_msgs

// @@protoc_insertion_point(global_scope)

#endif  // PROTOBUF_EscSensor_2eproto__INCLUDED
