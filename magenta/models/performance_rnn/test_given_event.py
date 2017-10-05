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


class pull_back_weight():

    """
    performance_lib.Performance

    """
    def __init__(self, Performance):
        self.performance = Performance


    def generate_specif_time(self, generator, time_step_z, args):
        """
        generate to the specific time of the event
        :param generator:
        :param self:
        :param time_step_z:
        :param args:
        :return:
        """
        MAX_SHIFT_STEPS = performance_lib.MAX_SHIFT_STEPS
        assert self.performance.num_steps < time_step_z
        print('free sample to time_step %d' % time_step_z)

        while self.performance.num_steps < time_step_z:
            total_steps = self.performance.__len__() + 1
            generator.initialize()
            self.performance, softmax_vec, indices = generator._model.generate_performance(
                total_steps, self.performance, **args)
            # generate 1 rnn_step

        value_shift = self.performance._events[-1].event_value
        pmf_prun = softmax_vec[-1][-MAX_SHIFT_STEPS:]
        fd = pmf_prun[value_shift - 1]
        Fd_denomin = sum(pmf_prun[value_shift:])
        self.performance.set_length(time_step_z)
        # trimmed probability, order of pitch (ascending or descending)
        w = np.log((fd*1000) / (Fd_denomin*1000))
        # * 1000, in case fd or Fd is too small
        return w


    def generate_fixed_onpitch(self, generator, pitch_z, args):
        """
        after arriving the specific time, turn on (the pitch of) the next given event
        :param generator:
        :param self:
        :param pitch_z:
        :param args:
        :return:
        """
        total_steps = self.performance.__len__() + 1
        generator.initialize()
        self.performance, softmax_vec, indices = generator._model.generate_performance(
            total_steps, self.performance, **args)
        fd = softmax_vec[-1][pitch_z]
        pitch_given = PerformanceEvent(event_type=1, event_value=pitch_z)
        self.performance.append(pitch_given)
        w = np.log(fd)

        return w


    def generate_unfixed_onpitch(self, generator, pitch_z, args):
        """
        after arriving the specific time, sample to(turn on) the specific given pitch
        assuming when turn on the pitch in ascending order
        :param generator:
        :param self:
        :param args:
        :return:
        """
        MAX_MIDI_PITCH = performance_lib.MAX_MIDI_PITCH - 1
        MAX_SHIFT_STEPS = performance_lib.MAX_SHIFT_STEPS
        hotcoding = PerformanceOneHotEncoding()
        time_step_z = self.performance.num_steps
        # time_step_z don't change with performance, array also won't change
        while self.performance.num_steps <= time_step_z:
            # run across time or pitch
            total_steps = self.performance.__len__() + 1
            generator.initialize()
            self.performance, softmax_vec, indices = generator._model.generate_performance(
                total_steps, self.performance, **args)
            encode_value = hotcoding.encode_event(self.performance._events[-1])

            if pitch_z < encode_value < MAX_MIDI_PITCH:
                break

        # pull back
        self.performance._events.pop()
        pitch_given = PerformanceEvent(event_type=1, event_value=pitch_z)
        self.performance.append(pitch_given)
        assert time_step_z == self.performance.num_steps

        pmf_prun_time = softmax_vec[-1][-MAX_SHIFT_STEPS:]
        pmf_prun_pitch = softmax_vec[-1][pitch_z + 1: MAX_MIDI_PITCH]
        fd = softmax_vec[-1][pitch_z]
        Fd_denomin = sum(pmf_prun_time) + sum(pmf_prun_pitch)
        w = np.log(fd / Fd_denomin)

        return w


    def generate_off_pitch(self, generator, pitch_z, time_on_z, args):
        """
        after arriving the specific time, turn off the specific note
        in order,,,,,,pitch, start time............
        :param generator:
        :param self:
        :param pitch_z:
        :param args:
        :return:
        """
        corrupt = self.check_duration(self, pitch_z, time_on_z)
        if corrupt:
            w = -10**(9)
            # if there are off and on in duration, give a very little value
        else:
            total_steps = self.performance.__len__() + 1
            generator.initialize()
            self.performance, softmax_vec, indices = generator._model.generate_performance(
                total_steps, self.performance, **args)
            fd = softmax_vec[-1][pitch_z]
            # just one matrix
            pitch_given = PerformanceEvent(event_type=2, event_value=pitch_z)
            self.performance.append(pitch_given)
            w = np.log(fd)

        return w


    def check_duration(self, pitch_z, time_on_z):
        """
        check whether there are turn on or turn off in the duration of one specific note
        :param pitch_z:
        :param time_on_z:
        :return:
        """
        j = 0
        corrupt = False
        while self.performance.num_steps > time_on_z:
            j = j - 1
            if self.performance._events[j].event_value == pitch_z:
                corrupt = True
                break

        return corrupt


class resamping_with_given():

    def __init__(self, num_particles, args, num_velocity_bins=0):
        self.generator = get_generator_flag()
        self.perf_list = []
        self.w = []
        self.steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
        self.args = args
        self.num_particles = num_particles
        self.num_velocity_bins = num_velocity_bins


    # def convert_input(self, t_s, p_z, t_e, b):
    #     t_conb = [t_s, t_e]
    #     t_conb_sort = sort(t_conb)

    def res_start_step(self, t_s, p_z):

        performance_0 = performance_lib.Performance(
            steps_per_second=self.steps_per_second,
            start_step=0,
            num_velocity_bins=self.num_velocity_bins)
        assert t_s[0] >= 0
        # what if == 0-------------------------
        total_steps = t_s[0] * self.steps_per_second
        for i in range(self.num_particles):
            # primer_perfor = primer_performance_flag()
            primer_perfor = performance_0
            pull_time = pull_back_weight(primer_perfor)
            w_i = pull_time.generate_specif_time(self.generator, total_steps, self.args)
            self.w.append(w_i)
            self.perf_list.append(pull_time.performance)
        # self.perf_list = [performance_0 for j in range(self.num_particles)]
        self.systematic_resample()
        # sample freely to start time

        for i in range(num_particles):
            primer_perfor = self.perf_list[i]
            pull_pitch = pull_back_weight(primer_perfor)
            w_i = pull_pitch.generate_unfixed_onpitch(self.generator, p_z[0], self.args)
            self.w.append(w_i)
            self.perf_list.append(pull_pitch.performance)
        self.systematic_resample()
        # sample freely to pitch, after arriving the start time


    def res_with_bi(self, t_s, p_z, t_e, b):
        # assert dimensions are the same
        self.res_start_step(t_s, p_z)
        j = 0

        num_given_events = p_z.__len__()
        for i_s in range(num_given_events-1):
            if b[i_s]:
                if t_s[i_s+1] == t_s[i_s]:
                    for i in range(num_particles):
                        performance_i = pull_back_weight(re_per_list_on[i])
                        if b_i:
                            w_i = performance_i.generate_unfixed_onpitch(generator, pitch_z_s, args)
                        else:
                            w_i = performance_i.generate_fixed_onpitch(generator, pitch_z_s, args)
                        self.w.append(w_i)
                        self.perf_list.append(performance_i.performance)
                    self.systematic_resample()
                    i_s = i_s + 1
                else:

                    while t_s[i_s+1] > t_e[j]:
                        total_steps = t_e[j] * self.steps_per_second
                        for i in range(self.num_particles):
                            primer_perfor = self.perf_list[i]
                            pull_time = pull_back_weight(primer_perfor)
                            w_i = pull_time.generate_specif_time(self.generator, total_steps, self.args)
                            self.w.append(w_i)
                            self.perf_list.append(pull_time.performance)
                        self.systematic_resample()
                        j = j + 1
                        if t_s[i_s+1] < t_e[j]:
                            break
                            # -------------------------------------------------
                    total_steps = t_e[j] * self.steps_per_second
                    for i in range(self.num_particles):
                        primer_perfor = self.perf_list[i]
                        pull_time = pull_back_weight(primer_perfor)
                        w_i = pull_time.generate_specif_time(self.generator, total_steps, self.args)
                        self.w.append(w_i)
                        self.perf_list.append(pull_time.performance)
                    # self.perf_list = [performance_0 for j in range(self.num_particles)]
                    self.systematic_resample()










    # def resampling_t(self):
    #     total_steps = t_s[0] * self.steps_per_second
    #     assert t_s[0] >= 0
    #     if t_s[0] > 0:
    #         for i in range(self.num_particles):
    #             primer_perfor = self.perf_list[i]
    #             performance_i = pull_back_weight(primer_perfor)
    #             w_i = performance_i.generate_specif_time(self.generator, total_steps, self.args)
    #             self.w.append(w_i)
    #             self.perf_list.append(performance_i.performance)

    def systematic_resample(self):
        """
        resampling based on the weights: simply duplicate and delete particles
        param: w, weights or log likely hood of weights, "list"
        return: a, the select index of particles, start from 0
        """
        w = np.array(self.w)
        # 1*n ndarray
        if min(w) < 0:
            w = np.exp(w)
            # if weight are log likely hood, converted it into normal format
        w = (w*1000) / (np.sum(w)*1000)
        # * 1000, in case the elements of w are too small
        print(w)
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

        self.perf_list = [self.perf_list[i] for i in re_index]
        self.w = []
        return re_index


def write_music_flag(performance, time_gen_libo):
    """
    Make the generate request num_outputs times and save the output as midi
    files.
    give weight a small value if the duration is more than 5 seconds.!!!!!!!!!!!!!!!!!
    if some notes don't have corresponding off. !!!!!!!!!!!!!!!!!!!!!!!!!!!!
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
    print('Wrote %d MIDI files to %s' % (1, output_dir))
    # tf.logging.info('Wrote %d MIDI files to %s' % (1, output_dir))




total_steps = FLAGS.num_steps
print('first given note on %d' % (total_steps))
off_steps = 2 * total_steps
pitch_z = 30
pitch_z_s = 60
b_i = 1
num_particles = FLAGS.num_outputs
args = {
    'temperature': FLAGS.temperature,
    'beam_size': FLAGS.beam_size,
    'branch_factor': FLAGS.branch_factor,
    'steps_per_iteration': FLAGS.steps_per_iteration
}
generator = get_generator_flag()
w_time = []

perf_list_t =[]
for i in range(num_particles):
    primer_perfor = primer_performance_flag()
    performance_i = pull_back_weight(primer_perfor)
    w_i = performance_i.generate_specif_time(generator, total_steps, args)
    # w_i, performance_i = generate_specif_time(generator, primer_perfor, total_steps, args)
    w_time.append(w_i)
    perf_list_t.append(performance_i.performance)

re_index_t = systematic_resample(w_time)
re_per_list_t = [perf_list_t[i] for i in re_index_t]


w_on = []
perf_list_on =[]
for i in range(num_particles):
    performance_i = pull_back_weight(re_per_list_t[i])
    w_i = performance_i.generate_unfixed_onpitch(generator, pitch_z, args)
    w_on.append(w_i)
    perf_list_on.append(performance_i.performance)

re_index_on = systematic_resample(w_on)
re_per_list_on = [perf_list_on[i] for i in re_index_on]


w_on = []
perf_list_on = []
for i in range(num_particles):
    performance_i = pull_back_weight(re_per_list_on[i])
    if b_i:
        w_i = performance_i.generate_unfixed_onpitch(generator, pitch_z_s, args)
    else:
        w_i = performance_i.generate_fixed_onpitch(generator, pitch_z_s, args)
    w_on.append(w_i)
    perf_list_on.append(performance_i.performance)

re_index_on = systematic_resample(w_on)
re_per_list_on_sa = [perf_list_on[i] for i in re_index_on]
# b = np.array([3, 2, 1, 0])
# ccc = [a[j] for j in b]

w_time = []
perf_list_t =[]
for i in range(num_particles):
    performance_i = pull_back_weight(re_per_list_on_sa[i])
    w_i = performance_i.generate_specif_time(generator, off_steps, args)
    # w_i, performance_i = generate_specif_time(generator, primer_perfor, total_steps, args)
    w_time.append(w_i)
    perf_list_t.append(performance_i.performance)

re_index_t = systematic_resample(w_time)
re_per_list_t = [perf_list_t[i] for i in re_index_t]

w_off = []
perf_list_off =[]
for i in range(num_particles):
    performance_i = pull_back_weight(re_per_list_t[i])
    w_i = performance_i.generate_fixed_onpitch(generator, pitch_z, args)
    w_off.append(w_i)
    perf_list_off.append(performance_i.performance)

re_index_off = systematic_resample(w_off)
re_per_list_off = [perf_list_off[i] for i in re_index_off]


time_gen_libo = float(re_per_list_off[0].num_steps)/performance_lib.DEFAULT_STEPS_PER_SECOND
write_music_flag(re_per_list_off[0], time_gen_libo)
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
