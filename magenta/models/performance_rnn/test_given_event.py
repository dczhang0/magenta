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
    different options to do pull back action
    """
    def __init__(self, Performance):
        self.performance = Performance


    def generate_specif_time(self, generator, time_step_z, args):
        """
        freely sample to the specific time of the event
        :param generator: generate settings
        :param time_step_z: the specific time step
        :param args: generate options
        :return: weight
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

        value_last_shift = self.performance._events[-1].event_value
        value_back_shift = np.int(time_step_z - (self.performance.num_steps - value_last_shift))
        pmf_prun = softmax_vec[-1][-MAX_SHIFT_STEPS:]
        fd = pmf_prun[value_back_shift-1]
        Fd_denomin = sum(pmf_prun[value_back_shift:])
        self.performance.set_length(np.int(time_step_z))
        # trimmed probability, order of pitch (ascending or descending)
        w = np.log((fd*1000) / (Fd_denomin*1000))
        # * 1000, in case fd or Fd is too small
        return w


    def shift_fix_time(self, generator, shift_step_z, args):
        """
        jump given time step
        what if bigger than performance_lib.MAX_SHIFT_STEPS
        :param generator: generate options
        :param shift_step_z: the time step of this jump
        :param args: generate options
        :return: weight
        """
        MAX_SHIFT_STEPS = performance_lib.MAX_SHIFT_STEPS
        index_shift = np.int(-MAX_SHIFT_STEPS - 1 + shift_step_z)
        total_steps = self.performance.__len__() + 1
        generator.initialize()
        self.performance, softmax_vec, indices = generator._model.generate_performance(
            total_steps, self.performance, **args)
        fd = softmax_vec[-1][index_shift]
        shift_given = PerformanceEvent(event_type=3, event_value=np.int(shift_step_z))
        self.performance.append(shift_given)
        w = np.log(fd)

        return w


    def generate_fixed_onpitch(self, generator, pitch_z, args):
        """
        after arriving the specific time, turn on (the pitch of) the next given event
        :param generator: generate options
        :param pitch_z: the pitch of given event
        :param args: generate options
        :return: weight
        """
        total_steps = self.performance.__len__() + 1
        generator.initialize()
        self.performance, softmax_vec, indices = generator._model.generate_performance(
            total_steps, self.performance, **args)
        fd = softmax_vec[-1][pitch_z]
        pitch_given = PerformanceEvent(event_type=1, event_value=np.int(pitch_z))
        self.performance.append(pitch_given)
        w = np.log(fd)

        return w


    def generate_unfixed_onpitch(self, generator, pitch_z, args):
        """
        after arriving the specific time, sample to(turn on) the specific given pitch
        assuming when turn on the pitch in ascending order
        :param generator: generate options
        :param pitch_z: the pitch of given event
        :param args: generate options
        :return: weight
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
        pitch_given = PerformanceEvent(event_type=1, event_value=np.int(pitch_z))
        self.performance.append(pitch_given)
        assert time_step_z == self.performance.num_steps

        pmf_prun_time = softmax_vec[-1][-MAX_SHIFT_STEPS:]
        pmf_prun_pitch = softmax_vec[-1][pitch_z + 1: MAX_MIDI_PITCH]
        fd = softmax_vec[-1][pitch_z]
        Fd_denomin = sum(pmf_prun_time) + sum(pmf_prun_pitch)
        w = np.log(fd / Fd_denomin)

        return w


    def generate_off_pitch(self, generator, pitch_z, args, time_on_z=-1):
        """
        after arriving the specific time, turn off the specific note
        in order,,,,,,pitch, start time............
        :param generator: generate options
        :param pitch_z: pitch of given event
        :param args: generate options
        :param time_on_z: the on time of given pitch
        :return: weight
        """
        if time_on_z > 0:
            corrupt = self.check_duration(self, pitch_z, time_on_z)
        else:
            corrupt = False

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
            pitch_given = PerformanceEvent(event_type=2, event_value=np.int(pitch_z))
            self.performance.append(pitch_given)
            w = np.log(fd)

        return w


    def check_duration(self, pitch_z, time_on_z):
        """
        check whether there are turn on or turn off in the duration of one specific note
        :param pitch_z: pitch of given event
        :param time_on_z: start time of given event
        :return: whether on or off appears in the duration
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
    """
    resampling with given inputs
    inputs are: T, P, E, B
    """


    def __init__(self, generator, num_particles, args, num_velocity_bins=0):
        self.generator = generator
        self.perf_list = []
        self.w = []
        self.steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
        self.args = args
        self.num_particles = num_particles
        self.num_velocity_bins = num_velocity_bins


    def res_start_step(self, T_z, P_z, E_z, B_z):

        performance_0 = performance_lib.Performance(
            steps_per_second=self.steps_per_second,
            start_step=0,
            num_velocity_bins=self.num_velocity_bins)
        assert T_z[0] >= 0
        # what if == 0-------------------------
        total_steps = np.int(T_z[0] * self.steps_per_second)
        performance_0.set_length(min(performance_lib.MAX_SHIFT_STEPS, total_steps))
        for i_t in range(self.num_particles):
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
            # if E_z[0]:
            #     w_i = pull_pitch.generate_unfixed_onpitch(self.generator, P_z[0], self.args)
            # else:
            #     w_i = pull_pitch.generate_off_pitch(self.generator, P_z[0], self.args)
            w_i = pull_pitch.generate_unfixed_onpitch(self.generator, P_z[0], self.args)
            self.w.append(w_i)
            self.perf_list[i] = pull_pitch.performance
        self.systematic_resample()
        # sample freely to pitch, after arriving the start time


    def res_with_given_actions(self, T_z, P_z, E_z, B_z):
        """
        resampling after the first option
        last element in B_z
        :param T_z: time
        :param P_z: pitch
        :param E_z: 1: on, 0 off
        :param B_z: indicator
        :return:
        """
        self.res_start_step(T_z, P_z, E_z, B_z)
        # print(self.perf_list[0]._events)
        num_given_events = T_z.__len__()
        for i_st in range(num_given_events - 1):
            total_steps = np.int(T_z[i_st+1] * self.steps_per_second)
            for i_t in range(self.num_particles):
                primer_perfor = self.perf_list[i_t]
                pull_time = pull_back_weight(primer_perfor)
                if B_z[i_st]:
                    w_i = pull_time.generate_specif_time(self.generator, total_steps, self.args)
                else:
                    shift_steps = total_steps - T_z[i_st] * self.steps_per_second
                    w_i = pull_time.shift_fix_time(self.generator, shift_steps, self.args)
                self.w.append(w_i)
                self.perf_list[i_t] = pull_time.performance
            self.systematic_resample()
            for i_p in range(self.num_particles):
                primer_perfor = self.perf_list[i_p]
                # w_i = pull_pitch.generate_fixed_onpitch(generator, pitch_z_s, args)
                pull_pitch = pull_back_weight(primer_perfor)
                if B_z[i_st]:
                    if E_z[i_st+1]:
                        w_i = pull_pitch.generate_unfixed_onpitch(self.generator, P_z[i_st+1], self.args)
                    else:
                        w_i = pull_pitch.generate_off_pitch(self.generator, P_z[i_st+1], self.args)
                else:
                    if E_z[i_st+1]:
                        w_i = pull_pitch.generate_fixed_onpitch(self.generator, P_z[i_st+1], self.args)
                    else:
                        w_i = pull_pitch.generate_off_pitch(self.generator, P_z[i_st+1], self.args)
                self.w.append(w_i)
                self.perf_list[i_p] = pull_pitch.performance
            self.systematic_resample()
            # print(i_st)


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
        w = (w*1000) / (np.sum(w*1000))
        # * 1000, in case the elements of w are too small
        print(w)
        print(sum(w))
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


    def write_music(self, output_dir, MAX_NOTE_DURATION_SECONDS, time_gen_libo=0):
        """
        Make the generate request num_outputs times and save the output as midi
        files.
        give weight a small value if the duration is more than 5 seconds.!!!!!!!!!!!!!!!!!
        if some notes don't have corresponding turn off. !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        :param output_dir: output directory
        :param MAX_NOTE_DURATION_SECONDS
        :param time_gen_libo: specificed time
        :return:
        """
        num_outputs = self.num_particles
        if not output_dir:
            tf.logging.fatal('--output_dir required')
            return

        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
        digits = len(str(num_outputs))
        # time_gen_libo = float(FLAGS.num_steps)/performance_lib.DEFAULT_STEPS_PER_SECOND

        if not tf.gfile.Exists(output_dir):
            tf.gfile.MakeDirs(output_dir)

        for i in range(num_outputs):
            performance = self.perf_list[i]
            generated_sequence = performance.to_sequence(
                max_note_duration=MAX_NOTE_DURATION_SECONDS)
            if time_gen_libo > 0:
                assert (generated_sequence.total_time - time_gen_libo) <= 1e-5
            midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
            midi_path = os.path.join(output_dir, midi_filename)
            magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)
        print('Wrote %d MIDI files to %s' % (num_outputs, output_dir))
            # tf.logging.info('Wrote %d MIDI files to %s' % (1, output_dir))


MAX_NOTE_DURATION_SECONDS = 5.0
num_particles = FLAGS.num_outputs
args = {
    'temperature': FLAGS.temperature,
    'beam_size': FLAGS.beam_size,
    'branch_factor': FLAGS.branch_factor,
    'steps_per_iteration': FLAGS.steps_per_iteration
}
output_dir = os.path.expanduser(FLAGS.output_dir)
generator = get_generator_flag()
T_z = [1.5, 2, 3, 4]
P_z = [50, 50, 40, 40]
E_z = [1, 0, 1, 0]
B_z = [0, 0, 0, 0]
ressampl = resamping_with_given(generator, num_particles, args)
ressampl.res_with_given_actions(T_z, P_z, E_z, B_z)
ressampl.write_music(output_dir, MAX_NOTE_DURATION_SECONDS)




# time_gen_libo = float(ressampl.perf_list[0].num_steps)/performance_lib.DEFAULT_STEPS_PER_SECOND
# write_music_flag(ressampl.perf_list[0], time_gen_libo)
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
