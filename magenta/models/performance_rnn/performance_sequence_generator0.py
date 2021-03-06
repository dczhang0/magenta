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
"""Performance RNN generation code as a SequenceGenerator interface."""

from functools import partial
import math

# internal imports

import tensorflow as tf

from magenta.models.performance_rnn import performance_lib
from magenta.models.performance_rnn import performance_model

import magenta.music as mm

import numpy as np
from magenta.models.performance_rnn.performance_lib import PerformanceEvent
import copy
# This model can leave hanging notes. To avoid cacophony we turn off any note
# after 5 seconds.
MAX_NOTE_DURATION_SECONDS = 5.0


class PerformanceRnnSequenceGenerator(mm.BaseSequenceGenerator):
  """Performance RNN generation code as a SequenceGenerator interface."""

  def __init__(self, model, details,
               steps_per_second=performance_lib.DEFAULT_STEPS_PER_SECOND,
               num_velocity_bins=0, max_note_duration=MAX_NOTE_DURATION_SECONDS,
               fill_generate_section=True, checkpoint=None, bundle=None):
    """Creates a PerformanceRnnSequenceGenerator.

    Args:
      model: Instance of PerformanceRnnModel.
      details: A generator_pb2.GeneratorDetails for this generator.
      steps_per_second: Number of quantized steps per second.
      num_velocity_bins: Number of quantized velocity bins. If 0, don't use
          velocity.
      max_note_duration: The maximum note duration in seconds to allow during
          generation. This model often forgets to release notes; specifying a
          maximum duration can force it to do so.
      fill_generate_section: If True, the model will generate RNN steps until
          the entire generate section has been filled. If False, the model will
          estimate the number of RNN steps needed and then generate that many
          events, even if the generate section isn't completely filled.
      checkpoint: Where to search for the most recent model checkpoint. Mutually
          exclusive with `bundle`.
      bundle: A GeneratorBundle object that includes both the model checkpoint
          and metagraph. Mutually exclusive with `checkpoint`.
    """
    super(PerformanceRnnSequenceGenerator, self).__init__(
        model, details, checkpoint, bundle)
    self.steps_per_second = steps_per_second
    self.num_velocity_bins = num_velocity_bins
    self.max_note_duration = max_note_duration
    self.fill_generate_section = fill_generate_section

  def _generate(self, input_sequence, generator_options):
    if len(generator_options.input_sections) > 1:
      raise mm.SequenceGeneratorException(
          'This model supports at most one input_sections message, but got %s' %
          len(generator_options.input_sections))
    if len(generator_options.generate_sections) != 1:
      raise mm.SequenceGeneratorException(
          'This model supports only 1 generate_sections message, but got %s' %
          len(generator_options.generate_sections))

    generate_section = generator_options.generate_sections[0]
    if generator_options.input_sections:
      input_section = generator_options.input_sections[0]
      primer_sequence = mm.trim_note_sequence(
          input_sequence, input_section.start_time, input_section.end_time)
      input_start_step = mm.quantize_to_step(
          input_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
    else:
      primer_sequence = input_sequence
      input_start_step = 0

    last_end_time = (max(n.end_time for n in primer_sequence.notes)
                     if primer_sequence.notes else 0)
    if last_end_time > generate_section.start_time:
      raise mm.SequenceGeneratorException(
          'Got GenerateSection request for section that is before or equal to '
          'the end of the NoteSequence. This model can only extend sequences. '
          'Requested start time: %s, Final note end time: %s' %
          (generate_section.start_time, last_end_time))

    # Quantize the priming sequence.
    quantized_primer_sequence = mm.quantize_note_sequence_absolute(
        primer_sequence, self.steps_per_second)

    extracted_perfs, _ = performance_lib.extract_performances(
        quantized_primer_sequence, start_step=input_start_step,
        num_velocity_bins=self.num_velocity_bins)
    assert len(extracted_perfs) <= 1

    generate_start_step = mm.quantize_to_step(
        generate_section.start_time, self.steps_per_second, quantize_cutoff=0.0)
    # libo-----------------start_time* steps_per_sceond +1----------------------------

    # Note that when quantizing end_step, we set quantize_cutoff to 1.0 so it
    # always rounds down. This avoids generating a sequence that ends at 5.0
    # seconds when the requested end time is 4.99.
    generate_end_step = mm.quantize_to_step(
        generate_section.end_time, self.steps_per_second, quantize_cutoff=1.0)

    if extracted_perfs and extracted_perfs[0]:
      performance = extracted_perfs[0]
    else:
      # If no track could be extracted, create an empty track that starts at the
      # requested generate_start_step.
      performance = performance_lib.Performance(
          steps_per_second=(
              quantized_primer_sequence.quantization_info.steps_per_second),
          start_step=generate_start_step,
          num_velocity_bins=self.num_velocity_bins)

    # Ensure that the track extends up to the step we want to start generating.
    # performance.set_length(generate_start_step - performance.start_step)
    # libo----------------------delete the function of "add a (3,1)--------------------------------

    # Extract generation arguments from generator options.
    arg_types = {
        'temperature': lambda arg: arg.float_value,
        'beam_size': lambda arg: arg.int_value,
        'branch_factor': lambda arg: arg.int_value,
        'steps_per_iteration': lambda arg: arg.int_value
    }
    args = dict((name, value_fn(generator_options.args[name]))
                for name, value_fn in arg_types.items()
                if name in generator_options.args)

    total_steps = performance.num_steps + (
        generate_end_step - generate_start_step+1)
    # libo---------------------- add +1 to stay the same with deletete------------------------------
    # num_steps: Returns how many steps long this sequence is the
    #   Length of the sequence in quantized steps.

    if not performance:
      # Primer is empty; let's just start with silence.
      performance.set_length(min(performance_lib.MAX_SHIFT_STEPS, total_steps))

    len_prim_events = len(performance)
    # libo--------------------------- the length of primer performance-------------------------------

    softmax_Libo = np.zeros((len_prim_events, 356)) - 2
    indices_Libo = np.zeros((len_prim_events, 1), dtype=np.int) - 2
    # libo-------------------generate arguments as long as the primer sequence---------------------------
    # libo-------negative number indicates primer sequence, 356 is the length of one-hot vector----------

    while performance.num_steps < total_steps:
      # Assume there's around 10 notes per second and 4 RNN steps per note.
      # Can't know for sure until generation is finished because the number of
      # notes per quantized step is variable.
      steps_to_gen = total_steps - performance.num_steps
      if steps_to_gen < 40:
          rnn_steps_to_gen = int(0.4 * steps_to_gen) + 1
      else:
          rnn_steps_to_gen = 40 * int(math.ceil(
          float(steps_to_gen) / performance_lib.DEFAULT_STEPS_PER_SECOND))
      tf.logging.info(
          'Need to generate %d more steps for this sequence, will try asking '
          'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
      performance, softmax_Libo_tmp, indices_Libo_tmp = self._model.generate_performance(
          len(performance) + rnn_steps_to_gen, performance, **args)
      # libo-----------------output more arguments------------------------------
      softmax_Libo = np.vstack((softmax_Libo, softmax_Libo_tmp))
      indices_Libo = np.vstack((indices_Libo, indices_Libo_tmp))


      if not self.fill_generate_section:
        # In the interest of speed just go through this loop once, which may not
        # entirely fill the generate section.
        break

    assert indices_Libo.__len__() == performance.__len__()
    # pitch 0-127 (indicate 1-128),, shift: 1-100
    # performance_raw_Libo=performance
    # performance_raw_Libo = copy.deepcopy(performance)
    # print('length of the whole sequence: %d' % (performance.__len__()))
    # performance.set_length(total_steps)
    # # since performance.num_steps > total_steps (time_steps, absolute time),
    # # prune it into total_steps, the _events steps decrease correspondingly.
    # print('length of the pruned whole sequence: %d' % (performance.__len__()))
    # print('length of primer sequence: %d' % (len_prim_events))

    # len_whol_seq_Libo = indices_Libo[0].__len__() + extracted_perfs[0]._events.__len__()
    # len_prun_seq_Libo = len_whol_seq_Libo - performance.__len__()
    # indices_prun_Libo = indices_Libo[0][0:-len_prun_seq_Libo + 1]
    # softmax_prun_Libo = softmax_Libo[0][0:-len_prun_seq_Libo + 1]
    #
    # aaa = PerformanceEvent(event_type=1, event_value=100)
    # performance.append(aaa)
    # # performance[-2:-1]
    # # performance[-2:]?????????????????????????--------------------

    # # PerformanceOneHotEncoding
    # while performance._events[-1].event_type != 3:
    #     performance._events.pop()

    # len_mag_Libo = performance.__len__()
    # l_libo = performance._events[-1].event_value
    # pmf_prun = softmax_Libo[len_mag_Libo - 1][-100:]
    # fd_Libo = pmf_prun[l_libo-1]
    # Fd_nominato = sum(pmf_prun[l_libo:])
    # #  this trimmed probability
    # w = fd_Libo / Fd_nominato

    # generated_sequence = performance.to_sequence(
    #     max_note_duration=self.max_note_duration)
    #
    # assert (generated_sequence.total_time - generate_section.end_time) <= 1e-5
    return performance, softmax_Libo, indices_Libo
    # original, return generated_sequence

  def generate_with_performance(self, performance, total_steps, args):
    #   generate with performance type of data
    if not performance:
        # Primer is empty; let's just start with silence.
        performance.set_length(min(performance_lib.MAX_SHIFT_STEPS, total_steps))

    len_prim_seq_Libo = performance._events.__len__()
    softmax_Libo = np.zeros((len_prim_seq_Libo, 356))
    indices_Libo = np.zeros((len_prim_seq_Libo, 1), dtype=np.int)

    while performance.num_steps < total_steps:
        # Assume there's around 10 notes per second and 4 RNN steps per note.
        # Can't know for sure until generation is finished because the number of
        # notes per quantized step is variable.
        steps_to_gen = total_steps - performance.num_steps
        if steps_to_gen < 40:
            rnn_steps_to_gen = int(0.4 * steps_to_gen) + 1
        else:
            rnn_steps_to_gen = 40 * int(math.ceil(
                float(steps_to_gen) / performance_lib.DEFAULT_STEPS_PER_SECOND))
        tf.logging.info(
            'Need to generate %d more steps for this sequence, will try asking '
            'for %d RNN steps' % (steps_to_gen, rnn_steps_to_gen))
        performance, softmax_Libo_tmp, indices_Libo_tmp = self._model.generate_performance(
            len(performance) + rnn_steps_to_gen, performance, **args)
        softmax_Libo = np.vstack((softmax_Libo, softmax_Libo_tmp))
        indices_Libo = np.vstack((indices_Libo, indices_Libo_tmp))

        if not self.fill_generate_section:
            # In the interest of speed just go through this loop once, which may not
            # entirely fill the generate section.
            break

    assert indices_Libo.__len__() == performance.__len__()
    return performance, softmax_Libo, indices_Libo

  # def generate_performance_rnnstep(self, total_steps, primer_perfor, args):
  #     self.initialize()
  #     primer_perfor, softmax_Libo, indices_Libo = self._model.generate_performance(
  #         total_steps, primer_perfor, **args)
  #     return primer_perfor, softmax_Libo, indices_Libo

def get_generator_map():
  """Returns a map from the generator ID to a SequenceGenerator class creator.

  Binds the `config` argument so that the arguments match the
  BaseSequenceGenerator class constructor.

  Returns:
    Map from the generator ID to its SequenceGenerator class creator with a
    bound `config` argument.
  """
  def create_sequence_generator(config, **kwargs):
    return PerformanceRnnSequenceGenerator(
        performance_model.PerformanceRnnModel(config), config.details,
        steps_per_second=config.steps_per_second,
        num_velocity_bins=config.num_velocity_bins, fill_generate_section=False,
        **kwargs)

  return {key: partial(create_sequence_generator, config)
          for (key, config) in performance_model.default_configs.items()}
