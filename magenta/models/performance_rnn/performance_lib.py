# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for working with polyphonic performances."""

from __future__ import division

import collections
import math

# internal imports
import tensorflow as tf

from magenta.music import constants
from magenta.music import events_lib
from magenta.music import sequences_lib
from magenta.pipelines import statistics
from magenta.protobuf import music_pb2

MAX_MIDI_PITCH = constants.MAX_MIDI_PITCH
MIN_MIDI_PITCH = constants.MIN_MIDI_PITCH

MAX_MIDI_VELOCITY = constants.MAX_MIDI_VELOCITY
MIN_MIDI_VELOCITY = constants.MIN_MIDI_VELOCITY
MAX_NUM_VELOCITY_BINS = MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1

STANDARD_PPQ = constants.STANDARD_PPQ

DEFAULT_STEPS_PER_SECOND = 100
MAX_SHIFT_STEPS = 1000
# MAX_SHIFT_STEPS = 100
# max_shift equal to "max_note_duration" in "to_sequence", which is the maximum note duration in generation.
# Libo-------------------------------------maximum shift step------500???------------------------------------
MAX_EVENTS = 512
# from magenta.models.performance_rnn.performance_rnn_create_dataset import MAX_EVENTS
# Libo----------trivial-------to reduce memory use, especially when the performance is too long--------------

class PerformanceEvent(object):
  """Class for storing events in a performance."""

  # Start of a new note.
  NOTE_ON = 1
  # End of a note.
  NOTE_OFF = 2
  # Shift time forward.
  TIME_SHIFT = 3
  # Change current velocity.
  VELOCITY = 4

  def __init__(self, event_type, event_value):
    if not PerformanceEvent.NOTE_ON <= event_type <= PerformanceEvent.VELOCITY:
      raise ValueError('Invalid event type: %s' % event_type)

    if (event_type == PerformanceEvent.NOTE_ON or
        event_type == PerformanceEvent.NOTE_OFF):
      if not MIN_MIDI_PITCH <= event_value <= MAX_MIDI_PITCH:
        raise ValueError('Invalid pitch value: %s' % event_value)
    elif event_type == PerformanceEvent.TIME_SHIFT:
      if not 1 <= event_value <= MAX_SHIFT_STEPS:
        raise ValueError('Invalid time shift value: %s' % event_value)
    elif event_type == PerformanceEvent.VELOCITY:
      if not 1 <= event_value <= MAX_NUM_VELOCITY_BINS:
        raise ValueError('Invalid velocity value: %s' % event_value)

    self.event_type = event_type
    self.event_value = event_value

  def __repr__(self):
    return 'PerformanceEvent(%r, %r)' % (self.event_type, self.event_value)

  def __eq__(self, other):
    if not isinstance(other, PerformanceEvent):
      return False
    return (self.event_type == other.event_type and
            self.event_value == other.event_value)


class Performance(events_lib.EventSequence):
  """Stores a polyphonic sequence as a stream of performance events.

  Events are PerformanceEvent objects that encode event type and value.
  """

  def __init__(self, quantized_sequence=None, steps_per_second=None,
               start_step=0, num_velocity_bins=0):
    """Construct a Performance.

    Either quantized_sequence or steps_per_second should be supplied.

    Args:
      quantized_sequence: A quantized NoteSequence proto.
      steps_per_second: Number of quantized time steps per second.
      start_step: The offset of this sequence relative to the
          beginning of the source sequence. If a quantized sequence is used as
          input, only notes starting after this step will be considered.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.

    Raises:
      ValueError: If `num_velocity_bins` is larger than the number of MIDI
          velocity values.
    """
    if (quantized_sequence, steps_per_second).count(None) != 1:
      raise ValueError(
          'Must specify exactly one of quantized_sequence or steps_per_second')

    if num_velocity_bins > MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1:
      raise ValueError(
          'Number of velocity bins is too large: %d' % num_velocity_bins)

    if quantized_sequence:
      sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)
      self._events = self._from_quantized_sequence(
          quantized_sequence, start_step, num_velocity_bins)
      self._steps_per_second = (
          quantized_sequence.quantization_info.steps_per_second)
    else:
      self._events = []
      self._steps_per_second = steps_per_second

    self._start_step = start_step
    self._num_velocity_bins = num_velocity_bins

  @property
  def start_step(self):
    return self._start_step

  @property
  def steps_per_second(self):
    return self._steps_per_second

  def _append_steps(self, num_steps):
    """Adds steps to the end of the sequence."""
    if (self._events and
        self._events[-1].event_type == PerformanceEvent.TIME_SHIFT and
        self._events[-1].event_value < MAX_SHIFT_STEPS):
      # Last event is already non-maximal time shift. Increase its duration.
      added_steps = min(num_steps,
                        MAX_SHIFT_STEPS - self._events[-1].event_value)
      self._events[-1] = PerformanceEvent(
          PerformanceEvent.TIME_SHIFT,
          self._events[-1].event_value + added_steps)
      num_steps -= added_steps

    while num_steps >= MAX_SHIFT_STEPS:
      self._events.append(
          PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                           event_value=MAX_SHIFT_STEPS))
      num_steps -= MAX_SHIFT_STEPS

    if num_steps > 0:
      self._events.append(
          PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                           event_value=num_steps))

  def _trim_steps(self, num_steps):
    """Trims a given number of steps from the end of the sequence."""
    steps_trimmed = 0
    while self._events and steps_trimmed < num_steps:
      if self._events[-1].event_type == PerformanceEvent.TIME_SHIFT:
        if steps_trimmed + self._events[-1].event_value > num_steps:
          self._events[-1] = PerformanceEvent(
              event_type=PerformanceEvent.TIME_SHIFT,
              event_value=(self._events[-1].event_value -
                           num_steps + steps_trimmed))
          steps_trimmed = num_steps
        else:
          steps_trimmed += self._events[-1].event_value
          self._events.pop()
      else:
        self._events.pop()

  def set_length(self, steps, from_left=False):
    """Sets the length of the sequence to the specified number of steps.

    If the event sequence is not long enough, pads with time shifts to make the
    sequence the specified length. If it is too long, it will be truncated to
    the requested length.

    Args:
      steps: How many quantized steps long the event sequence should be.
      from_left: Whether to add/remove from the left instead of right.
    """
    if from_left:
      raise NotImplementedError('from_left is not supported')

    if self.num_steps < steps:
      self._append_steps(steps - self.num_steps)
    elif self.num_steps > steps:
      self._trim_steps(self.num_steps - steps)

    assert self.num_steps == steps

  def append(self, event):
    """Appends the event to the end of the sequence.

    Args:
      event: The performance event to append to the end.

    Raises:
      ValueError: If `event` is not a valid performance event.
    """
    if not isinstance(event, PerformanceEvent):
      raise ValueError('Invalid performance event: %s' % event)
    self._events.append(event)

  def truncate(self, num_events):
    """Truncates this Performance to the specified number of events.

    Args:
      num_events: The number of events to which this performance will be
          truncated.
    """
    self._events = self._events[:num_events]

  def __len__(self):
    """How many events are in this sequence.

    Returns:
      Number of events as an integer.
    """
    return len(self._events)

  def __getitem__(self, i):
    """Returns the event at the given index."""
    return self._events[i]

  def __iter__(self):
    """Return an iterator over the events in this sequence."""
    return iter(self._events)

  def __str__(self):
    strs = []
    for event in self:
      if event.event_type == PerformanceEvent.NOTE_ON:
        strs.append('(%s, ON)' % event.event_value)
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        strs.append('(%s, OFF)' % event.event_value)
      elif event.event_type == PerformanceEvent.TIME_SHIFT:
        strs.append('(%s, SHIFT)' % event.event_value)
      elif event.event_type == PerformanceEvent.VELOCITY:
        strs.append('(%s, VELOCITY)' % event.event_value)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)
    return '\n'.join(strs)

  @property
  def end_step(self):
    return self.start_step + self.num_steps

  @property
  def num_steps(self):
    """Returns how many steps long this sequence is.

    Returns:
      Length of the sequence in quantized steps.
    """
    steps = 0
    for event in self:
      if event.event_type == PerformanceEvent.TIME_SHIFT:
        steps += event.event_value
    return steps

  @staticmethod
  def _from_quantized_sequence(quantized_sequence, start_step=0,
                               num_velocity_bins=0):
    """Populate self with events from the given quantized NoteSequence object.

    Within a step, new pitches are started with NOTE_ON and existing pitches are
    ended with NOTE_OFF. TIME_SHIFT shifts the current step forward in time.
    VELOCITY changes the current velocity value that will be applied to all
    NOTE_ON events.

    Args:
      quantized_sequence: A quantized NoteSequence instance.
      start_step: Start converting the sequence at this time step.
      num_velocity_bins: Number of velocity bins to use. If 0, velocity events
          will not be included at all.

    Returns:
      A list of events.
    """
    notes = [note for note in quantized_sequence.notes
             if not note.is_drum and note.quantized_start_step >= start_step]
    sorted_notes = sorted(notes, key=lambda note: note.start_time)

    # Sort all note start and end events.
    onsets = [(note.quantized_start_step, idx, False)
              for idx, note in enumerate(sorted_notes)]
    offsets = [(note.quantized_end_step, idx, True)
               for idx, note in enumerate(sorted_notes)]
    note_events = sorted(onsets + offsets)

    if num_velocity_bins:
      velocity_bin_size = int(math.ceil(
          (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) / num_velocity_bins))
      velocity_to_bin = (
          lambda v: (v - MIN_MIDI_VELOCITY) // velocity_bin_size + 1)

    current_step = start_step
    current_velocity_bin = 0
    performance_events = []

    if len(note_events) > 0:
      note_events = Performance.actions_ascending_pitch(sorted_notes, note_events)
      # print('i am okay')
    # else:
    #   print('i am not okay')
    # Libo---------------------for order of pitch-------------------add----------
    for step, idx, is_offset in note_events:
      # if step > current_step:
      #   # Shift time forward from the current step to this event.
      #   while step > current_step + MAX_SHIFT_STEPS:
      #     # We need to move further than the maximum shift size.
      #     performance_events.append(
      #         PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
      #                          event_value=MAX_SHIFT_STEPS))
      #     current_step += MAX_SHIFT_STEPS
      #   performance_events.append(
      #       PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
      #                        event_value=int(step - current_step)))
      #   current_step = step
      num_event_step = len(performance_events)
      shift_step = step - current_step
      if shift_step > MAX_SHIFT_STEPS or num_event_step > MAX_EVENTS:
        # Shift time forward from the current step to this event.
        if shift_step > MAX_SHIFT_STEPS:
          print('shift steps %s >1000 or len events %s > 512' % (shift_step, num_event_step))
        return performance_events
      elif step > current_step:
        performance_events.append(
          PerformanceEvent(event_type=PerformanceEvent.TIME_SHIFT,
                           event_value=int(shift_step)))
        current_step = step
      # Libo------------------- maximum shift step(time)----------------------revise--------
      # Libo--------number of events stop criteria to save memory ------------revise--------

      # If we're using velocity and this note's velocity is different from the
      # current velocity, change the current velocity.
      if num_velocity_bins:
        velocity_bin = velocity_to_bin(sorted_notes[idx].velocity)
        if not is_offset and velocity_bin != current_velocity_bin:
          current_velocity_bin = velocity_bin
          performance_events.append(
              PerformanceEvent(event_type=PerformanceEvent.VELOCITY,
                               event_value=current_velocity_bin))

      # Add a performance event for this note on/off.
      event_type = (PerformanceEvent.NOTE_OFF if is_offset
                    else PerformanceEvent.NOTE_ON)
      performance_events.append(
          PerformanceEvent(event_type=event_type,
                           event_value=sorted_notes[idx].pitch))

    return performance_events


  @staticmethod
  def actions_ascending_pitch(sorted_notes, note_events):
    """
    find the actions at the same time step, rearrange the events with serial order:
    first on then off, ascending pitch
    :param sorted_notes: sequence of notes
    :param note_events: sequence of actions
    :return: filtered sequence
    """
    sorted_actions = []
    st_same = 1
    # print('size of sorted_actions %s' % len(sorted_actions))
    sorted_actions.append(note_events[0])
    i = 1
    # sorted_actions.append((note_events[0], sorted_notes[note_events[0][1]].pitch))
    for step, idx, is_offset in note_events[1:]:
      sorted_actions.append((step, idx, is_offset))
      i = i + 1
      last_step = sorted_actions[-2][0]
      if last_step == step:
        st_same = st_same + 1
        if len(note_events) == i:
          sorted_actions = Performance.ascending_pitch(sorted_actions, st_same, sorted_notes)
          # assert len(sorted_actions) == len(note_events)
          return sorted_actions
      elif st_same > 1:
        sorted_actions.pop()
        sorted_actions = Performance.ascending_pitch(sorted_actions, st_same, sorted_notes)
        sorted_actions.append((step, idx, is_offset))
        st_same = 1
      # print(st_same)
    # assert len(sorted_actions) == len(note_events)
    return sorted_actions
  # Libo---------------------for order of pitch-------------------add----------

  @staticmethod
  def ascending_pitch(sorted_actions, st_same, sorted_notes):
    """
    rearrange the actions with ascending pitch, and [on, off, shift]
    # 127(note on)+127(note off) +100 (shift)
    :param sorted_actions: actions sequence to be rearranged
    :param st_same:  number of actions to be rearranged, from the reverse direction
    :param sorted_notes: sequence of notes
    :return: filtered sequence of actions
    """
    # print('i am not okay')
    # sorted([2, 1, 3, 4, 5], key=lambda x: (x < 3, x))
    actions_on_p = []
    actions_off_p = []
    pitch_off = []
    pitch_on =[]
    # pitch = 0
    for i in range(st_same):
      # if pitch == sorted_notes[sorted_actions[-1][1]].pitch:
      #   sorted_actions.pop()
      #   continue
      pitch = sorted_notes[sorted_actions[-1][1]].pitch
      # sorted_actions[-1][1]: -1 indicate the last element in list, [1] indicate the idx in sorted notes
      if sorted_actions[-1][2] and (pitch not in pitch_off):
        pitch_off.append(pitch)
        actions_off_p.append((sorted_actions[-1], pitch))
      elif (not sorted_actions[-1][2]) and (pitch not in pitch_on):
        pitch_on.append(pitch)
        actions_on_p.append((sorted_actions[-1], pitch))
      sorted_actions.pop()

    actions_on_p = sorted(actions_on_p, key=lambda x: x[1])
    # for i in range(len(actions_on_p)-1):
    #   if actions_on_p[i][1] == actions_on_p[i+1][1]:
    #     # print(actions_on_p[i])
    #     del actions_on_p[i]
    actions_off_p = sorted(actions_off_p, key=lambda x: x[1])
    # for i in range(len(actions_off_p)-1):
    #   if actions_off_p[i][1] == actions_off_p[i+1][1]:
    #     # print(actions_off_p[i])
    #     del actions_off_p[i]
    actions_on = [actions_on_p[i][0] for i in range(len(actions_on_p))]
    actions_off = [actions_off_p[i][0] for i in range(len(actions_off_p))]
    sorted_actions = sorted_actions + actions_on + actions_off
    # [note on, note off, shift]
    # [1, 2, 3] + [1, 2, 3]
    # for i in range(num_last_notes):
    #   pitch_ser.append(notes_to_filter[-i-1].pitch)
    return sorted_actions
  # Libo---------------------for order of pitch-------------------add----------

  def to_sequence(self,
                  velocity=100,
                  instrument=0,
                  program=0,
                  max_note_duration=None):
    """Converts the Performance to NoteSequence proto.

    Args:
      velocity: MIDI velocity to give each note. Between 1 and 127 (inclusive).
          If the performance contains velocity events, those will be used
          instead.
      instrument: MIDI instrument to give each note.
      program: MIDI program to give each note.
      max_note_duration: Maximum note duration in seconds to allow. Notes longer
          than this will be truncated. If None, notes can be any length.

    Raises:
      ValueError: if an unknown event is encountered.

    Returns:
      A NoteSequence proto.
    """
    seconds_per_step = 1.0 / self._steps_per_second

    sequence_start_time = self.start_step * seconds_per_step

    sequence = music_pb2.NoteSequence()
    sequence.ticks_per_quarter = STANDARD_PPQ

    step = 0

    if self._num_velocity_bins:
      velocity_bin_size = int(math.ceil(
          (MAX_MIDI_VELOCITY - MIN_MIDI_VELOCITY + 1) /
          self._num_velocity_bins))

    # Map pitch to list because one pitch may be active multiple times.
    pitch_start_steps_and_velocities = collections.defaultdict(list)
    for i, event in enumerate(self):
      if event.event_type == PerformanceEvent.NOTE_ON:
        pitch_start_steps_and_velocities[event.event_value].append(
            (step, velocity))
      elif event.event_type == PerformanceEvent.NOTE_OFF:
        if not pitch_start_steps_and_velocities[event.event_value]:
          tf.logging.debug(
              'Ignoring NOTE_OFF at position %d with no previous NOTE_ON' % i)
        else:
          # Create a note for the pitch that is now ending.
          pitch_start_step, pitch_velocity = pitch_start_steps_and_velocities[
              event.event_value][0]
          pitch_start_steps_and_velocities[event.event_value] = (
              pitch_start_steps_and_velocities[event.event_value][1:])
          if step == pitch_start_step:
            tf.logging.debug(
                'Ignoring note with zero duration at step %d' % step)
            continue
          note = sequence.notes.add()
          note.start_time = (pitch_start_step * seconds_per_step +
                             sequence_start_time)
          note.end_time = step * seconds_per_step + sequence_start_time
          if (max_note_duration and
              note.end_time - note.start_time > max_note_duration):
            # print(note.end_time, note.start_time)
            #libo-------------------maximum shift step = max_note_duration---------------------
            note.end_time = note.start_time + max_note_duration
          note.pitch = event.event_value
          note.velocity = pitch_velocity
          note.instrument = instrument
          note.program = program
          if note.end_time > sequence.total_time:
            sequence.total_time = note.end_time
      elif event.event_type == PerformanceEvent.TIME_SHIFT:
        step += event.event_value
      elif event.event_type == PerformanceEvent.VELOCITY:
        assert self._num_velocity_bins
        velocity = (
            MIN_MIDI_VELOCITY + (event.event_value - 1) * velocity_bin_size)
      else:
        raise ValueError('Unknown event type: %s' % event.event_type)

    # There could be remaining pitches that were never ended. End them now
    # and create notes.
    for pitch in pitch_start_steps_and_velocities:
      for pitch_start_step, pitch_velocity in pitch_start_steps_and_velocities[
          pitch]:
        if step == pitch_start_step:
          tf.logging.debug(
              'Ignoring note with zero duration at step %d' % step)
          continue
        note = sequence.notes.add()
        note.start_time = (pitch_start_step * seconds_per_step +
                           sequence_start_time)
        note.end_time = step * seconds_per_step + sequence_start_time
        if (max_note_duration and
            note.end_time - note.start_time > max_note_duration):
          note.end_time = note.start_time + max_note_duration
        note.pitch = pitch
        note.velocity = pitch_velocity
        note.instrument = instrument
        note.program = program
        if note.end_time > sequence.total_time:
          sequence.total_time = note.end_time

    return sequence


def extract_performances(
    quantized_sequence, start_step=0, min_events_discard=None,
    max_events_truncate=None, num_velocity_bins=0):
  """Extracts a performance from the given quantized NoteSequence.

  Currently, this extracts only one performance from a given track.

  Args:
    quantized_sequence: A quantized NoteSequence.
    start_step: Start extracting a sequence at this time step.
    min_events_discard: Minimum length of tracks in events. Shorter tracks are
        discarded.
    max_events_truncate: Maximum length of tracks in events. Longer tracks are
        truncated.
    num_velocity_bins: Number of velocity bins to use. If 0, velocity events
        will not be included at all.

  Returns:
    performances: A python list of Performance instances.
    stats: A dictionary mapping string names to `statistics.Statistic` objects.
  """
  sequences_lib.assert_is_absolute_quantized_sequence(quantized_sequence)

  stats = dict([(stat_name, statistics.Counter(stat_name)) for stat_name in
                ['performances_discarded_too_short',
                 'performances_truncated',
                 'performances_discarded_more_than_1_program']])

  steps_per_second = quantized_sequence.quantization_info.steps_per_second

  # Create a histogram measuring lengths (in bars not steps).
  stats['performance_lengths_in_seconds'] = statistics.Histogram(
      'performance_lengths_in_seconds',
      [5, 10, 20, 30, 40, 60, 120])

  # Allow only 1 program.
  programs = set()
  for note in quantized_sequence.notes:
    programs.add(note.program)
  if len(programs) > 1:
    stats['performances_discarded_more_than_1_program'].increment()
    return [], stats.values()

  performances = []

  # Translate the quantized sequence into a Performance.
  performance = Performance(quantized_sequence, start_step=start_step,
                            num_velocity_bins=num_velocity_bins)
  # ---------------libo:Performance.validation_of_order(performance)-----------------

  if (max_events_truncate is not None and
      len(performance) > max_events_truncate):
    performance.truncate(max_events_truncate)
    stats['performances_truncated'].increment()

  if min_events_discard is not None and len(performance) < min_events_discard:
    stats['performances_discarded_too_short'].increment()
  else:
    performances.append(performance)
    stats['performance_lengths_in_seconds'].increment(
        performance.num_steps // steps_per_second)
  # Libo--------------------important information------------------record the time of the performances----------
  return performances, stats.values()
