import ast
import os
import time

# internal imports

import tensorflow as tf
import magenta

from magenta.models.performance_rnn import performance_model
from magenta.models.performance_rnn import performance_sequence_generator

from magenta.music import constants
from magenta.protobuf import generator_pb2
from magenta.protobuf import music_pb2

from config_libo import *
from magenta.models.performance_rnn import performance_lib
from magenta.models.performance_rnn.performance_lib import PerformanceEvent
from magenta.models.performance_rnn.performance_encoder_decoder import PerformanceOneHotEncoding
import numpy as np
# import test_files_xin

MAX_NOTE_DURATION_SECONDS = 5.0

def run_with_flags():
  """
  Generates performance tracks, Uses the options specified by the flags from config_libo
  :return, the sequence list "performance",
  Args:
    generator: The PerformanceRnnSequenceGenerator to use for generation.
  """
  tf.logging.set_verbosity(FLAGS.log)

  bundle = get_bundle()

  config_id = bundle.generator_details.id if bundle else FLAGS.config
  config = performance_model.default_configs[config_id]
  #????performance_lib.DEFAULT_STEPS_PER_SECOND,
  config.hparams.parse(FLAGS.hparams)
  # Having too large of a batch size will slow generation down unnecessarily.
  config.hparams.batch_size = min(
      config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)

  generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
      model=performance_model.PerformanceRnnModel(config),
      details=config.details,
      steps_per_second=config.steps_per_second,
      num_velocity_bins=config.num_velocity_bins,
      checkpoint=get_checkpoint(),
      bundle=bundle)

  if not FLAGS.output_dir:
    tf.logging.fatal('--output_dir required')
    return
  output_dir = os.path.expanduser(FLAGS.output_dir)

  primer_midi = None
  if FLAGS.primer_midi:
    primer_midi = os.path.expanduser(FLAGS.primer_midi)

  if not tf.gfile.Exists(output_dir):
    tf.gfile.MakeDirs(output_dir)

  primer_sequence = None
  if FLAGS.primer_pitches:
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
    for pitch in ast.literal_eval(FLAGS.primer_pitches):
      note = primer_sequence.notes.add()
      note.start_time = 0
      note.end_time = 60.0 / magenta.music.DEFAULT_QUARTERS_PER_MINUTE
      #--------------------????????????????????-------------------
      note.pitch = pitch
      note.velocity = 100
      primer_sequence.total_time = note.end_time
  elif FLAGS.primer_melody:
    primer_melody = magenta.music.Melody(ast.literal_eval(FLAGS.primer_melody))
    #  Melody: extract monophonic melodies-----------------------
    primer_sequence = primer_melody.to_sequence()
    # melodies_lib: Converts the Melody to NoteSequence proto.
  elif primer_midi:
    primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi)
  else:
    tf.logging.warning(
        'No priming sequence specified. Defaulting to empty sequence.')
    primer_sequence = music_pb2.NoteSequence()
    primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ

  print(primer_sequence)
  # primer_sequence.notes
  # Derive the total number of seconds to generate.
  seconds_per_step = 1.0 / generator.steps_per_second
  generate_end_time = FLAGS.num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  # Set the start time to begin when the last note ends.
  generate_section = generator_options.generate_sections.add(
      start_time=primer_sequence.total_time,
      end_time=generate_end_time)

  if generate_section.start_time >= generate_section.end_time:
    tf.logging.fatal(
        'Priming sequence is longer than the total number of steps '
        'requested: Priming sequence length: %s, Total length '
        'requested: %s',
        generate_section.start_time, generate_end_time)
    return

  generator_options.args['temperature'].float_value = FLAGS.temperature
  generator_options.args['beam_size'].int_value = FLAGS.beam_size
  generator_options.args['branch_factor'].int_value = FLAGS.branch_factor
  generator_options.args[
      'steps_per_iteration'].int_value = FLAGS.steps_per_iteration

  tf.logging.debug('primer_sequence: %s', primer_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  # for i in range(FLAGS.num_outputs):
  performance, softmax_Libo, indices_Libo = generator.generate(primer_sequence, generator_options)

    # sequence_generator: generate
    # Performance_Sequence_Generator: _generate
    # performance_model: generate_performance
    # events_rnn_model: _generate_events

  return performance, softmax_Libo, indices_Libo


def write_music(performance):
    """
    Make the generate request num_outputs times and save the output as midi
    files.
    :param performance:
    :return: midi file
    """

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(FLAGS.num_outputs))
    output_dir = FLAGS.output_dir
    time_gen_libo = FLAGS.num_steps/performance_lib.DEFAULT_STEPS_PER_SECOND
    generated_sequence = performance.to_sequence(
        max_note_duration=MAX_NOTE_DURATION_SECONDS)
    assert (generated_sequence.total_time - time_gen_libo) <= 1e-5
    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)
    tf.logging.info('Wrote %d MIDI files to %s', 1, output_dir)


def weights_trunc_sequence():
    MAX_SHIFT_STEPS = performance_lib.MAX_SHIFT_STEPS
    performance, softmax_Libo, indices_Libo = run_with_flags()
    # performance[-2:-1]
    # performance[-2:]?????????????????????????--------------------
    # def _trim_steps(self, num_steps):

    while performance._events[-1].event_type != 3:
        performance._events.pop()

    len_mag_Libo = performance.__len__()
    value_libo = performance._events[-1].event_value
    pmf_prun = softmax_Libo[len_mag_Libo - 1][-MAX_SHIFT_STEPS:]
    fd_Libo = pmf_prun[value_libo - 1]
    Fd_nominato = sum(pmf_prun[value_libo:])
    # trimmed probability
    w_i = fd_Libo / Fd_nominato
    return w_i, performance


def systematic_resample(w):
    """
    resampling based on the weights: simply duplicate and delete particles
    param: w, weights, "list"
    return: a, the select index of particles, start from 0
    """
    w = w / np.sum(w)
    n = len(w)
    u = np.random.rand() / n
    s = w[0]
    j = 0
    re_index = np.zeros(n, dtype=int)
    ninv = 1 / n
    for k in range(n):
        while s < u:
            j += 1
            s += w[j]
        re_index[k] = j
        u += ninv
    return re_index


w = []
perf_list =[]
for i in range(FLAGS.num_outputs):
    w_i, performance_i = weights_trunc_sequence()
    w.append(w_i)
    perf_list.append(performance_i)

re_index = systematic_resample(w)
re_per_list = [perf_list[j] for j in re_index]
# a = [1, 2, 3, 4]
# b = np.array([3, 2, 1, 0])
# ccc = [a[j] for j in b]

write_music(performance_i)
# aaa = PerformanceEvent(event_type=1, event_value=100)
# performance.append(aaa)
#
# hotcoding = PerformanceOneHotEncoding()
# hotcoding.decode_event(100)
