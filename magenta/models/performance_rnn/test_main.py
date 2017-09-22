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

def generate_sect_with_flags(primer_sequence, num_steps):
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

  print(primer_sequence)
  # primer_sequence.notes
  # Derive the total number of seconds to generate.
  seconds_per_step = 1.0 / generator.steps_per_second
  generate_end_time = num_steps * seconds_per_step

  # Specify start/stop time for generation based on starting generation at the
  # end of the priming sequence and continuing until the sequence is num_steps
  # long.
  generator_options = generator_pb2.GeneratorOptions()
  # Set the start time to begin when the last note ends.
  generate_section = generator_options.generate_sections.add(
      start_time=primer_sequence.total_time,
      end_time=generate_end_time)

  # generator_options1 = generator_pb2.GeneratorOptions()
  # generate_section1 = generator_options1.generate_sections.add(
  #     start_time=primer_sequence.total_time,
  #     end_time=2)

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
  generator_options.args['steps_per_iteration'].int_value = FLAGS.steps_per_iteration

  tf.logging.debug('primer_sequence: %s', primer_sequence)
  tf.logging.debug('generator_options: %s', generator_options)

  performance, softmax_Libo, indices_Libo = generator.generate(primer_sequence, generator_options)
    # sequence_generator: generate
    # Performance_Sequence_Generator: _generate
    # performance_model: generate_performance
    # events_rnn_model: _generate_events

  return performance, softmax_Libo, indices_Libo


def write_music(performance, time_gen_libo):
    """
    Make the generate request num_outputs times and save the output as midi
    files.
    :param performance:
    :return: midi file
    """
    if not FLAGS.output_dir:
        tf.logging.fatal('--output_dir required')
        return

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(FLAGS.num_outputs))
    # output_dir = FLAGS.output_dir
    # time_gen_libo = float(FLAGS.num_steps)/performance_lib.DEFAULT_STEPS_PER_SECOND
    output_dir = os.path.expanduser(FLAGS.output_dir)

    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    generated_sequence = performance.to_sequence(
        max_note_duration=MAX_NOTE_DURATION_SECONDS)
    assert (generated_sequence.total_time - time_gen_libo) <= 1e-5
    midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
    midi_path = os.path.join(output_dir, midi_filename)
    magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)
    tf.logging.info('Wrote %d MIDI files to %s', 1, output_dir)


def weights_sect_sequence(performance, softmax_Libo, total_steps):
    MAX_SHIFT_STEPS = performance_lib.MAX_SHIFT_STEPS
    print('length of the whole sequence: %d' % (performance.__len__()))
    performance.set_length(total_steps)
    # since performance.num_steps > total_steps (time_steps, absolute time),
    # prune it into total_steps, the _events steps decrease correspondingly.

    while performance._events[-1].event_type != 3:
        performance._events.pop()
    print('length of the pruned whole sequence: %d' % (performance.__len__()))
    # turn off the note right before the beginning of the start-------------------
    len_mag_Libo = performance.__len__()
    value_libo = performance._events[-1].event_value
    pmf_prun = softmax_Libo[len_mag_Libo - 1][-MAX_SHIFT_STEPS:]
    fd_Libo = pmf_prun[value_libo - 1]
    Fd_denomin = sum(pmf_prun[value_libo:])
    # trimmed probability, order of pitch (ascending or descending)
    w_i = fd_Libo / Fd_denomin
    return w_i, performance
#---------------------------probability in the fommer sequence


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


def get_primer_sequence():

    primer_midi = None
    if FLAGS.primer_midi:
        primer_midi = os.path.expanduser(FLAGS.primer_midi)

    primer_sequence = None
    if FLAGS.primer_pitches:
        primer_sequence = music_pb2.NoteSequence()
        primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
        for pitch in ast.literal_eval(FLAGS.primer_pitches):
            note = primer_sequence.notes.add()
            note.start_time = 0
            note.end_time = 60.0 / magenta.music.DEFAULT_QUARTERS_PER_MINUTE
            # --------------------????????????????????-------------------
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

    return primer_sequence

def get_checkpoint():
  """Get the training dir or checkpoint path to be used by the model."""
  if FLAGS.run_dir and FLAGS.bundle_file and not FLAGS.save_generator_bundle:
    raise magenta.music.SequenceGeneratorException(
        'Cannot specify both bundle_file and run_dir')
  if FLAGS.run_dir:
    train_dir = os.path.join(os.path.expanduser(FLAGS.run_dir), 'train')
    return train_dir
  else:
    return None


def get_bundle():
  """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

  Returns:
    Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
    not set or the save_generator_bundle flag is set.
  """
  if FLAGS.save_generator_bundle:
    return None
  if FLAGS.bundle_file is None:
    return None
  bundle_file = os.path.expanduser(FLAGS.bundle_file)
  return magenta.music.read_bundle_file(bundle_file)

def generate_fixed_sect(primer_perfor, fixed_sect, args):
    """
    :param primer_perfor: performance type
    :param fixed_sect: performance type
    :param args: generate options
    :return: generated sequence and the weights of the generated sequence
    """
    bundle = get_bundle()
    config_id = bundle.generator_details.id if bundle else FLAGS.config
    config = performance_model.default_configs[config_id]
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

    w = 0
    total_steps = primer_perfor.__len__()
    fixed_sect_len = fixed_sect.__len__()

    for i in range(fixed_sect_len):
        total_steps = total_steps + 1
        # primer_perfor, softmax_Libo, indices_Libo = generator.generate_performance_rnnstep(
        #     total_steps, primer_perfor, args)
        generator.initialize()
        primer_perfor, softmax_Libo, indices_Libo = generator._model.generate_performance(
            total_steps, primer_perfor, **args)
        assert total_steps == primer_perfor.__len__()
        primer_perfor._events.pop()
        fixed_event_i = fixed_sect[i]
        primer_perfor.append(fixed_event_i)
        hotcoding = PerformanceOneHotEncoding()
        fixed_seq_value_i = hotcoding.encode_event(fixed_event_i)
        p = softmax_Libo[0, fixed_seq_value_i]
        w = w + np.log(p)
    return w, primer_perfor

w = []
perf_list =[]
total_steps = FLAGS.num_steps


for i in range(FLAGS.num_outputs):
    primer_sequence = get_primer_sequence()
    performance, softmax_Libo, indices_Libo = generate_sect_with_flags(primer_sequence, total_steps)
    # performance[-2:-1]
    # performance[-2:]?????????????????????????--------------------
    w_i, performance_i = weights_sect_sequence(performance, softmax_Libo, total_steps)
    w.append(w_i)
    perf_list.append(performance_i)

re_index = systematic_resample(w)
re_per_list = [perf_list[j] for j in re_index]
w = np.ones((re_index.shape))
w = w.tolist()
# a = [1, 2, 3, 4]
# b = np.array([3, 2, 1, 0])
# ccc = [a[j] for j in b]

args = {
    'temperature': FLAGS.temperature,
    'beam_size': FLAGS.beam_size,
    'branch_factor': FLAGS.branch_factor,
    'steps_per_iteration': FLAGS.steps_per_iteration
}
w, performance_after_fix = generate_fixed_sect(performance_i, performance_i, args)
# bundle = get_bundle()
#
# config_id = bundle.generator_details.id if bundle else FLAGS.config
# config = performance_model.default_configs[config_id]
# config.hparams.parse(FLAGS.hparams)
# # Having too large of a batch size will slow generation down unnecessarily.
# config.hparams.batch_size = min(
#     config.hparams.batch_size, FLAGS.beam_size * FLAGS.branch_factor)
#
# generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
#     model=performance_model.PerformanceRnnModel(config),
#     details=config.details,
#     steps_per_second=config.steps_per_second,
#     num_velocity_bins=config.num_velocity_bins,
#     checkpoint=get_checkpoint(),
#     bundle=bundle)
#
# performance, softmax_Libo, indices_Libo = generator.generate_performance(performance, total_steps, args)

time_gen_libo = float(total_steps)/performance_lib.DEFAULT_STEPS_PER_SECOND
write_music(performance_i, time_gen_libo)
# aaa = PerformanceEvent(event_type=1, event_value=100)
# performance.append(aaa)
#
# hotcoding = PerformanceOneHotEncoding()
# hotcoding.decode_event(100)