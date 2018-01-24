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

from magenta.models.performance_rnn.config_libo import *
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
import copy

class Constraints(object):

    def __init__(self, t, p, a, b):
        self.t = t
        self.p = p
        self.e = a
        self.b = b


    @classmethod
    def constraints_from_performance(cls, primer_performance):
        """
        event_type: 1: on; 2: off; 3: shift
        :param primer_performance:
        :return:
        """
        length = primer_performance.__len__()
        t = []
        p = []
        a = []
        b = []
        performance = copy.deepcopy(primer_performance)
        for i in range(length):
            if performance._events[-1].event_type != 3:
                t.append(performance.num_steps)
                a.append(performance._events[-1].event_type)
                p.append(performance._events[-1].event_value)
            performance._events.pop()
        # t = [t[-j - 1] for j in range(len(t))]
        # a = [a[-j - 1] for j in range(len(t))]
        # p = [p[-j - 1] for j in range(len(t))]
        t.reverse()
        a.reverse()
        p.reverse()

        # convert to 1 on, 0 off.
        zzz = 2 - np.array(a)
        a = np.list(zzz)
        constr = cls(t, p, a, b)

        return constr


    @staticmethod
    def valid(sequence):
        return True or False

# class A(object):
#     def foo(self,x):
#         print "executing foo(%s,%s)"%(self,x)
#
#     @classmethod
#     def class_foo(cls,x):
#         print "executing class_foo(%s,%s)"%(cls,x)
#
#     @staticmethod
#     def static_foo(x):
#         print "executing static_foo(%s)"%x

class weight_generate(object):
    """
    performance_lib.Performance
    different options to do pull back action
    """
    def __init__(self, Performance):
        self.performance = Performance


    def sample_to_time(self, generator, time_step_z, args):
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
            self.performance, softmax_vec = generator._model.generate_performance(
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


    def jump_to_time(self, generator, shift_step_z, args):
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
        self.performance, softmax_vec = generator._model.generate_performance(
            total_steps, self.performance, **args)
        fd = softmax_vec[-1][index_shift]
        shift_given = PerformanceEvent(event_type=3, event_value=np.int(shift_step_z))
        self.performance._events.pop()
        self.performance._events.append(shift_given)
        w = np.log(fd*1000) - np.log(1000)

        return w


    def sample_to_onpitch(self, generator, pitch_z, args):
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
            self.performance, softmax_vec = generator._model.generate_performance(
                total_steps, self.performance, **args)
            encode_value = hotcoding.encode_event(self.performance._events[-1])

            if pitch_z < encode_value < MAX_MIDI_PITCH:
                break

        # pull back
        self.performance._events.pop()
        pitch_given = PerformanceEvent(event_type=1, event_value=np.int(pitch_z))
        self.performance._events.append(pitch_given)
        assert time_step_z == self.performance.num_steps

        pmf_prun_time = softmax_vec[-1][-MAX_SHIFT_STEPS:]
        pmf_prun_pitch = softmax_vec[-1][pitch_z + 1: MAX_MIDI_PITCH]
        fd = softmax_vec[-1][pitch_z]
        Fd_denomin = sum(pmf_prun_time) + sum(pmf_prun_pitch)
        w = np.log(fd / Fd_denomin)

        return w


    def jump_to_onpitch(self, generator, pitch_z, args):
        """
        after arriving the specific time, turn on (the pitch of) the next given event
        :param generator: generate options
        :param pitch_z: the pitch of given event
        :param args: generate options
        :return: weight
        """
        total_steps = self.performance.__len__() + 1
        generator.initialize()
        self.performance, softmax_vec = generator._model.generate_performance(
            total_steps, self.performance, **args)
        self.performance._events.pop()
        pitch_given = PerformanceEvent(event_type=1, event_value=np.int(pitch_z))
        hotcoding = PerformanceOneHotEncoding()
        index = hotcoding.encode_event(pitch_given)
        fd = softmax_vec[-1][index]
        self.performance._events.append(pitch_given)
        w = np.log(fd)

        return w


    def jump_to_offpitch(self, generator, pitch_z, args):
        """
        after arriving the specific time, turn off the specific note
        in order,,,,,,pitch, start time............
        :param generator: generate options
        :param pitch_z: pitch of given event
        :param args: generate options
        :return: weight
        """
        total_steps = self.performance.__len__() + 1
        generator.initialize()
        self.performance, softmax_vec = generator._model.generate_performance(
            total_steps, self.performance, **args)
        self.performance._events.pop()
        pitch_given = PerformanceEvent(event_type=2, event_value=np.int(pitch_z))
        hotcoding = PerformanceOneHotEncoding()
        index = hotcoding.encode_event(pitch_given)
        fd = softmax_vec[-1][index]
        self.performance._events.append(pitch_given)
        w = np.log(fd)

        return w


class sampler_given_actions(object):
    """
    resampling with given inputs, write music to midi file
    inputs are: T, P, E, B
    """

    def __init__(self, generator, num_particles, args, num_velocity_bins=0):
        """

        :param generator:
        :param num_particles:
        :param args:
        :param num_velocity_bins:
        """
        self.generator = generator
        self.perf_list = []
        self.w = []
        self.steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
        self.args = args
        self.num_particles = num_particles
        self.num_velocity_bins = num_velocity_bins

    def sample(self, T_z, P_z, E_z, B_z):
        """
        sample with given actions: sample to the time of action, then sample to the given action
        last element in B_z
        :param T_z: time
        :param P_z: pitch
        :param E_z: 1: on, 0 off
        :param B_z: indicator
        :return:
        """
        self.sample_to_start(T_z, P_z, E_z, B_z)
        # print(self.perf_list[0]._events)
        num_given_events = T_z.__len__()
        for i_st in range(num_given_events - 1):

            total_steps = np.int(T_z[i_st + 1] * self.steps_per_second)
            for i_t in range(self.num_particles):
                primer_perfor = self.perf_list[i_t]
                pull_time = weight_generate(primer_perfor)
                if B_z[i_st]:
                    w_i = pull_time.sample_to_time(self.generator, total_steps, self.args)
                else:
                    shift_steps = total_steps - T_z[i_st] * self.steps_per_second
                    w_i = pull_time.jump_to_time(self.generator, shift_steps, self.args)
                self.w.append(w_i)
                self.perf_list[i_t] = pull_time.performance
            self.systematic_resample()

            for i_p in range(self.num_particles):
                primer_perfor = self.perf_list[i_p]
                # w_i = pull_pitch.generate_fixed_onpitch(generator, pitch_z_s, args)
                pull_pitch = weight_generate(primer_perfor)
                if B_z[i_st]:
                    if E_z[i_st + 1]:
                        w_i = pull_pitch.sample_to_onpitch(self.generator, P_z[i_st + 1], self.args)
                    else:
                        w_i = pull_pitch.jump_to_offpitch(self.generator, P_z[i_st + 1], self.args)
                else:
                    if E_z[i_st + 1]:
                        w_i = pull_pitch.jump_to_onpitch(self.generator, P_z[i_st + 1], self.args)
                    else:
                        w_i = pull_pitch.jump_to_offpitch(self.generator, P_z[i_st + 1], self.args)
                self.w.append(w_i)
                self.perf_list[i_p] = pull_pitch.performance
            self.systematic_resample()
            # print(i_st)

    def sample_to_start(self, T_z, P_z, E_z, B_z):

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
            pull_time = weight_generate(primer_perfor)
            w_i = pull_time.sample_to_time(self.generator, total_steps, self.args)
            self.w.append(w_i)
            self.perf_list.append(pull_time.performance)
        # self.perf_list = [performance_0 for j in range(self.num_particles)]
        self.systematic_resample()
        # sample freely to start time

        for i in range(self.num_particles):
            primer_perfor = self.perf_list[i]
            pull_pitch = weight_generate(primer_perfor)
            # if E_z[0]:
            #     w_i = pull_pitch.generate_unfixed_onpitch(self.generator, P_z[0], self.args)
            # else:
            #     w_i = pull_pitch.generate_off_pitch(self.generator, P_z[0], self.args)
            w_i = pull_pitch.sample_to_onpitch(self.generator, P_z[0], self.args)
            self.w.append(w_i)
            self.perf_list[i] = pull_pitch.performance
        self.systematic_resample()
        # sample freely to pitch, after arriving the start time

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
        w = (w * 1000) / (np.sum(w * 1000))
        # * 1000, in case the elements of w are too small
        print(w)
        print(sum(w))
        n = len(w)
        u = np.random.rand() / n
        s = w[0]
        j = 0
        re_index = np.zeros(n, dtype=int)
        ninv = float(1) / n  # or 1.0/n , different form python 3,
        for k in range(n):
            while s < u:
                j += 1
                s += w[j]
            re_index[k] = j
            u += ninv

        self.perf_list = [self.perf_list[i] for i in re_index]
        self.w = []
        return re_index

    def write_music(self, output_dir, max_note_duration, total_time=0):
        """
        Make the generate request num_outputs times and save the output as midi
        files.
        give weight a small value if the duration is more than 5 seconds.!!!!!!!!!!!!!!!!!
        if some notes don't have corresponding turn off. !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        :param output_dir: output directory
        :param max_note_duration
        :param total_time: specificed time
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
            generated_sequence = performance.to_sequence(max_note_duration=max_note_duration)

            if total_time > 0:
                assert (generated_sequence.total_time - total_time) <= 1e-5
            midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
            midi_path = os.path.join(output_dir, midi_filename)
            magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)
        print('Wrote %d MIDI files to %s' % (num_outputs, output_dir))
        # tf.logging.info('Wrote %d MIDI files to %s' % (1, output_dir))


class generator_bundle_args(object):

    def __init__(self, bundle_file, save_generator_bundle=False, run_dir=None):
        self.save_generator_bundle = save_generator_bundle
        self.bundle_file = bundle_file
        self.run_dir = run_dir
        self.bundle = self.get_bundle()
        self.checkpoint = self.get_checkpoint()


    def get_checkpoint(self):
      """Get the training dir or checkpoint path to be used by the model."""
      if self.run_dir and self.bundle_file and not self.save_generator_bundle:
        raise magenta.music.SequenceGeneratorException(
            'Cannot specify both bundle_file and run_dir')
      if self.run_dir:
        train_dir = os.path.join(os.path.expanduser(self.run_dir), 'train')
      else:
        train_dir = None
      return train_dir


    def get_bundle(self):
      """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

      Returns:
        Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
        not set or the save_generator_bundle flag is set.
      """
      if self.save_generator_bundle:
        return None
      if self.bundle_file is None:
        return None
      bundle_file = os.path.expanduser(self.bundle_file)
      bundle = magenta.music.read_bundle_file(bundle_file)
      return bundle


    def get_generator(self, config_name, beam_size=1, branch_factor=1, hparams=''):
        """
        config, hparams='', beam_size=1, branch_factor=1
        :param config:
        :param hparams:
        :param beam_size:
        :param branch_factor:
        :return:
        """
        bundle = self.bundle
        config_id = bundle.generator_details.id if bundle else config_name
        # bundle.generator_details.id: the name of file in the dictionary
        # config_id = np.unicode(FLAGS.config)
        config = performance_model.default_configs[config_id]
        config.hparams.parse(hparams)
        # Having too large of a batch size will slow generation down unnecessarily.
        config.hparams.batch_size = min(
            config.hparams.batch_size, beam_size * branch_factor)

        generator = performance_sequence_generator.PerformanceRnnSequenceGenerator(
            model=performance_model.PerformanceRnnModel(config),
            details=config.details,
            steps_per_second=config.steps_per_second,
            num_velocity_bins=config.num_velocity_bins,
            checkpoint=self.checkpoint,
            bundle=bundle)
        return generator


def extract_primer_performance(primer_pitches=None, primer_melody=None, primer_midi=None):
    """
    obtain the former performance format of data from path of midi file, sequence of primer pitches,
    sequence of primer melodies
    :param primer_pitches: A string representation of a Python list of pitches that will be used as
                    a starting chord with a short duration.  with a quarter note duration
    :param primer_melody: A string representation of a Python list of 'magenta.music.Melody' event values.
                    For example: "[60, -2, 60, -2, 67, -2, 67, -2]". The primer melody will be played at '
                    'a fixed tempo of 120 QPM with 4 steps per quarter note.
                    (-2 = no event, -1 = note-off event, values 0 through 127 = note-on event for that MIDI pitch)
                    'monophonic melodies'
     :param primer_midi: The path to a MIDI file containing a polyphonic track
    :return: performance sequence
    # def spam(a, b=None, c=None):
    #     print(b)
    #     print(c)
    # spam(100, c=0)
    # spam(100)
    """
    steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
    input_start_step = 0
    num_velocity_bins = 0

    primer_sequence = None
    if primer_pitches:
        primer_sequence = music_pb2.NoteSequence()
        primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ
        for pitch in ast.literal_eval(primer_pitches):
            note = primer_sequence.notes.add()
            note.start_time = 0
            note.end_time = 60.0 / magenta.music.DEFAULT_QUARTERS_PER_MINUTE
            # --------------------????????????????????-------------------
            note.pitch = pitch
            note.velocity = 100
            primer_sequence.total_time = note.end_time
    elif primer_melody:
        primer_melody = magenta.music.Melody(ast.literal_eval(primer_melody))
        primer_sequence = primer_melody.to_sequence()
        # melodies_lib: Converts the Melody to NoteSequence proto.
    elif primer_midi:
        primer_midi = os.path.expanduser(primer_midi)
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


def main(unused_argv):
    """
    :param unused_argv:
    :return:
    """
    primer_melody = FLAGS.primer_melody
    config = FLAGS.config
    bundle_file = FLAGS.bundle_file
    MAX_NOTE_DURATION_SECONDS = 5.0
    num_particles = FLAGS.num_outputs
    args = {
        'temperature': FLAGS.temperature,
        'beam_size': FLAGS.beam_size,
        'branch_factor': FLAGS.branch_factor,
        'steps_per_iteration': FLAGS.steps_per_iteration
    }
    output_dir = os.path.expanduser(FLAGS.output_dir)
    aaa = generator_bundle_args(bundle_file=bundle_file)
    generator = aaa.get_generator(config_name=config)
    primer_performance = extract_primer_performance(primer_melody=primer_melody)
    T_z = [15, 20, 30, 40]
    P_z = [50, 50, 40, 40]
    E_z = [1, 0, 1, 0]
    B_z = [0, 0, 0, 0]
    ressampl = sampler_given_actions(generator, num_particles, args)
    ressampl.sample(T_z, P_z, E_z, B_z)
    ressampl.write_music(output_dir, MAX_NOTE_DURATION_SECONDS)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()