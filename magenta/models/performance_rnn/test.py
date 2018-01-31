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
import copy
from concurrent import futures
from magenta.music import note_sequence_io
# from magenta.scripts.convert_dir_to_note_sequences import queue_conversions
# from magenta.scripts.convert_dir_to_note_sequences import convert_directory
#---------------------------------------import error----------------------------
from magenta.pipelines import pipeline
from magenta.protobuf.music_pb2 import NoteSequence
from magenta.music.sequences_lib import extract_subsequence
from magenta.models.shared.events_rnn_model import EventSequenceRnnModel

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

def onestep_forward_org(performance, generator):

    args = {
        'temperature': FLAGS.temperature,
        'beam_size': FLAGS.beam_size,
        'branch_factor': FLAGS.branch_factor,
        'steps_per_iteration': FLAGS.steps_per_iteration
    }

    total_steps = len(performance) + 1
    performance_org, softmax_org = generator._model.generate_performance(
        total_steps, performance, **args)
    return performance_org, softmax_org

def onestep_forward_new(performance, generator):

    event_sequences, softmax_ini, final_state_ini = generator._model.first_update(performance)
    indices = generator._model._config.encoder_decoder.extend_event_sequences(
        event_sequences, softmax_ini)
    return event_sequences, softmax_ini[0], final_state_ini

def melody_to_sequence(simple_melody, input_start_step=0, num_velocity_bins=0):
    primer_melody = magenta.music.Melody(ast.literal_eval(simple_melody))
    primer_sequence = primer_melody.to_sequence()
    steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
    quantized_primer_sequence = mm.quantize_note_sequence_absolute(primer_sequence, steps_per_second)

    extracted_perfs, _ = performance_lib.extract_performances(
        quantized_primer_sequence, start_step=input_start_step,
        num_velocity_bins=num_velocity_bins)
    performance = extracted_perfs[0]
    return performance

def main(unused_argv):
    """
    :param unused_argv:
    :return:
    """

    simple_melody = FLAGS.primer_melody
    performance = melody_to_sequence(simple_melody)
    end_time_steps = 3000

    config = FLAGS.config
    bundle_file = FLAGS.bundle_file
    aaa = generator_bundle_args(bundle_file=bundle_file)
    generator = aaa.get_generator(config_name=config)
    # initial step (first step test)
    generator.initialize()
    performance_org, softmax_org = onestep_forward_org(performance, generator)

    performance_new, softmax_new, _ = onestep_forward_new(performance, generator)
    # assert softmax_new.all() == softmax_org.all()
    # assert softmax_new.any() == softmax_org.any()
    assert np.array_equal(softmax_new, softmax_org), "generate with rnn state is not correct"
    print("okay")


def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()