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
# aaa = PerformanceEvent(event_type=3, event_value=100)
from magenta.models.performance_rnn.performance_encoder_decoder import PerformanceOneHotEncoding
# hotcoding = PerformanceOneHotEncoding()
# hotcoding.encode_event(aaa)
# hotcoding.decode_event(355)
import numpy as np
import magenta.music as mm
# import test_files_xin

MAX_NOTE_DURATION_SECONDS = 5.0


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


def get_generator_flag():
    bundle = get_bundle()

    config_id = bundle.generator_details.id if bundle else FLAGS.config
    # config_id = np.unicode(FLAGS.config)
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
    return generator


def primer_performance_flag():
    """
    obtain the former performance format of data from flag
    :return:
    """
    steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
    input_start_step = 0
    num_velocity_bins = 0

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

    quantized_primer_sequence = mm.quantize_note_sequence_absolute(primer_sequence, steps_per_second)
    extracted_perfs, _ = performance_lib.extract_performances(
        quantized_primer_sequence, start_step=input_start_step,
        num_velocity_bins=num_velocity_bins)
    performance = extracted_perfs[0]
    # might be empty if no input.

    return performance



def generate_specif_time(generator, primer_perfor, time_step_z, args):
    """
    generate to the specific time of the event
    :param generator:
    :param primer_perfor:
    :param time_step_z:
    :param args:
    :return:
    """
    MAX_SHIFT_STEPS = performance_lib.MAX_SHIFT_STEPS
    assert primer_perfor.num_steps < time_step_z

    while primer_perfor.num_steps < time_step_z:
        total_steps = primer_perfor.__len__() + 1
        generator.initialize()
        primer_perfor, softmax_Libo, indices_Libo = generator._model.generate_performance(
            total_steps, primer_perfor, **args)
        # generate 1 rnn_step

    value_shift = primer_perfor._events[-1].event_value
    pmf_prun = softmax_Libo[-1][-MAX_SHIFT_STEPS:]
    fd_Libo = pmf_prun[value_shift - 1]
    Fd_denomin = sum(pmf_prun[value_shift:])
    primer_perfor.set_length(time_step_z)
    # trimmed probability, order of pitch (ascending or descending)
    w = np.log(fd_Libo / Fd_denomin)
    return w, primer_perfor


def generate_fixed_onpitch(generator, primer_perfor, pitch_z, args):
    """
    after arriving the specific time, turn on (the pitch of) the next given event
    :param generator:
    :param primer_perfor:
    :param pitch_z:
    :param args:
    :return:
    """
    total_steps = primer_perfor.__len__() + 1
    generator.initialize()
    primer_perfor, softmax_Libo, indices_Libo = generator._model.generate_performance(
        total_steps, primer_perfor, **args)
    fd_Libo = softmax_Libo[-1][pitch_z]
    pitch_given = PerformanceEvent(event_type=1, event_value=pitch_z)
    primer_perfor.append(pitch_given)
    w = np.log(fd_Libo)

    return w, primer_perfor


def generate_unfixed_onpitch(generator, primer_perfor, pitch_z, time_step_z, args):
    """
    after arriving the specific time, sample to(turn on) the specific given pitch
    assuming when turn on the pitch in ascending order
    :param generator:
    :param primer_perfor:
    :param time_step_z:
    :param args:
    :return:
    """
    MAX_MIDI_PITCH = performance_lib.MAX_MIDI_PITCH - 1
    hotcoding = PerformanceOneHotEncoding()
    assert primer_perfor.num_steps == time_step_z

    while primer_perfor.num_steps == time_step_z:
        # run across time or pitch
        total_steps = primer_perfor.num_steps + 1
        generator.initialize()
        primer_perfor, softmax_Libo, indices_Libo = generator._model.generate_performance(
            total_steps, primer_perfor, **args)
        encode_value = hotcoding.encode_event(primer_perfor._events[-1])

        if pitch_z < encode_value < MAX_MIDI_PITCH:
            break

    # pull back
    primer_perfor._events.pop()
    pitch_given = PerformanceEvent(event_type=1, event_value=pitch_z)
    primer_perfor.append(pitch_given)

    pmf_prun_time = softmax_Libo[-1][-MAX_SHIFT_STEPS:]
    pmf_prun_pitch = softmax_Libo[-1][pitch_z + 1: MAX_MIDI_PITCH]
    fd_Libo = softmax_Libo[-1][pitch_z]
    Fd_denomin = sum(pmf_prun_time) + sum(pmf_prun_pitch)
    w = np.log(fd_Libo / Fd_denomin)

    return w, primer_perfor


def generate_off_pitch(generator, primer_perfor, pitch_z, args):
    """
    after arriving the specific time, turn off the specific note
    :param generator:
    :param primer_perfor:
    :param pitch_z:
    :param args:
    :return:
    """
    total_steps = primer_perfor.__len__() + 1
    generator.initialize()
    primer_perfor, softmax_Libo, indices_Libo = generator._model.generate_performance(
        total_steps, primer_perfor, **args)
    fd_Libo = softmax_Libo[-1][pitch_z]
    # just one matrix, consider the turn off or turn on in the duration!!!!!!!!!!!!!!!!!!
    pitch_given = PerformanceEvent(event_type=2, event_value=pitch_z)
    primer_perfor.append(pitch_given)
    w = np.log(fd_Libo)
    return w, primer_perfor


def systematic_resample(w):
    """
    resampling based on the weights: simply duplicate and delete particles
    param: w, weights or log likely hood of weights, "list"
    return: a, the select index of particles, start from 0
    """
    if w[0] < 0:
        w = np.exp(w)
        # if weight are log likely hood, converted it into normal format
    w = np.array(w)
    w = w / np.sum(w)
    n = len(w)
    u = np.random.rand() / n
    s = w[0]
    j = 0
    re_index = np.zeros(n, dtype=int)
    ninv = float(1) / n    # or 1.0/n , different form python 3,
    for k in range(n):
        while s < u:
            j += 1
            s += w[j]
        re_index[k] = j
        u += ninv
    return re_index


def write_music_flag(performance, time_gen_libo):
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




w_free = []
perf_list =[]
total_steps = FLAGS.num_steps
num_particles = FLAGS.num_outputs
args = {
    'temperature': FLAGS.temperature,
    'beam_size': FLAGS.beam_size,
    'branch_factor': FLAGS.branch_factor,
    'steps_per_iteration': FLAGS.steps_per_iteration
}


primer_perfor = primer_performance_flag()
generator = get_generator_flag()
for i in range(num_particles):
    w_i, performance_i = generate_specif_time(generator, primer_perfor, total_steps, args)
    w_free.append(w_i)
    perf_list.append(performance_i)

re_index = systematic_resample(w_free)
re_per_list = [perf_list[i] for i in re_index]


w_free = []
perf_list =[]
for i in range(num_particles):
    w_i, performance_i = generate_specif_time(generator, primer_perfor, total_steps, args)
    w_free.append(w_i)
    perf_list.append(performance_i)
re_index = systematic_resample(w_free)
re_per_list = [perf_list[i] for i in re_index]
# w_free = np.ones((re_index.shape))
# w_free = w_free.tolist()
# a = [1, 2, 3, 4]
# b = np.array([3, 2, 1, 0])
# ccc = [a[j] for j in b]



w_fixed = []
perf_fix_list =[]
for j in range(num_particles):
    w_fixed_i, performance_after_fix = generate_fixed_sect(re_per_list[j], performance_i, args)
    w_fixed.append(w_fixed_i)
    perf_fix_list.append(performance_after_fix)

re_fix_index = systematic_resample(w_fixed)
re_fix_per_list = [perf_fix_list[j] for j in re_fix_index]

time_gen_libo = float(re_fix_per_list[0].num_steps)/performance_lib.DEFAULT_STEPS_PER_SECOND
write_music_flag(re_fix_per_list[0], time_gen_libo)
# need for_loop

# aaa = PerformanceEvent(event_type=1, event_value=100)
# performance.append(aaa)
#
# hotcoding = PerformanceOneHotEncoding()
# hotcoding.decode_event(100)

# def extract_performances(
#     quantized_sequence, start_step=0, min_events_discard=None,
#     max_events_truncate=None, num_velocity_bins=0):
#     if (quantized_sequence, steps_per_second).count(None) != 1:
#       raise ValueError(
#           'Must specify exactly one of quantized_sequence or steps_per_second')


# generate_start_step = mm.quantize_to_step(start_time, steps_per_second, quantize_cutoff=0.0)
# # If no track could be extracted, create an empty track that starts at the
# # requested generate_start_step.
# performance = performance_lib.Performance(
#     steps_per_second=steps_per_second,
#     start_step=generate_start_step,
#     num_velocity_bins=num_velocity_bins)
