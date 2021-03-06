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
"""Create a dataset of SequenceExamples from NoteSequence protos.

This script will extract polyphonic tracks from NoteSequence protos and save
them to TensorFlow's SequenceExample protos for input to the performance RNN
models. It will apply data augmentation, stretching and transposing each
NoteSequence within a limited range.
"""

import os

# internal imports

import tensorflow as tf

from magenta.models.performance_rnn import performance_lib
from magenta.models.performance_rnn import performance_model

from magenta.music import encoder_decoder
from magenta.pipelines import dag_pipeline
from magenta.pipelines import note_sequence_pipelines
from magenta.pipelines import pipeline
from magenta.pipelines import pipelines_common
from magenta.protobuf import music_pb2
from magenta.models.performance_rnn.performance_encoder_decoder import PerformanceOneHotEncoding
import copy
MAX_EVENTS = 512
# as in main
# /tmp/performance_rnn/generated
# '/tmp/notesequences.tfrecord',
# '/home/zha231/Downloads/notesequences.tfrecord'
# '/home/zha231/Downloads/magenta-d700/magenta/testdata/notesequences.tfrecord'
# Libo---------------------trivial-------------------input and out -------------------------------

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', '~/data/notesequences11.tfrecord',
                           'TFRecord to read NoteSequence protos from.')
tf.app.flags.DEFINE_string('output_dir', '~/data/performance_rnn/sequence_examples',
                           'Directory to write training and eval TFRecord '
                           'files. The TFRecord files are populated with '
                           'SequenceExample protos.')
tf.app.flags.DEFINE_string('config', 'performance', 'The config to use')
tf.app.flags.DEFINE_float('eval_ratio', 0.1,
                          'Fraction of input to set aside for eval set. '
                          'Partition is randomly selected.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')


class PerformanceExtractor(pipeline.Pipeline):
  """Extracts polyphonic tracks from a quantized NoteSequence."""

  def __init__(self, min_events, max_events, num_velocity_bins, name=None):
    super(PerformanceExtractor, self).__init__(
        input_type=music_pb2.NoteSequence,
        output_type=performance_lib.Performance,
        name=name)
    self._min_events = min_events
    self._max_events = max_events
    self._num_velocity_bins = num_velocity_bins

  def transform(self, quantized_sequence):
    performances, stats = performance_lib.extract_performances(
        quantized_sequence,
        min_events_discard=self._min_events,
        max_events_truncate=self._max_events,
        num_velocity_bins=self._num_velocity_bins)
    # print(1)
    # print(quantized_sequence.id)
    for i in range(len(performances)):
        self.validation_of_order(performances[i])
    # Libo---------------------revise check order and shift--------------------------------------------------
    self._set_stats(stats)
    # print(performances.end_time)
    # assert len(performances) == 1
    # perfor = performances[0]
    return performances


  @staticmethod
  def validation_of_order(performance_sequence):
    """
    :param sequence:
    :return:
    """
    # assert isinstance(sequence, Performance)
    assert isinstance(performance_sequence, performance_lib.Performance), "not a Performance data"
    hotcoding = PerformanceOneHotEncoding()
    intege_seq = [hotcoding.encode_event(performance_sequence[j]) for j in range(len(performance_sequence))]
    # [on, off, shift]
    end_pitch = performance_lib.MAX_MIDI_PITCH*2 + 1
    for i in range(len(intege_seq))[1:]:
      if intege_seq[i] > end_pitch:
        assert intege_seq[i-1] <= end_pitch, "two shift, %d" % i
      elif intege_seq[i-1] <= end_pitch:
          # if intege_seq[i] == intege_seq[i-1]:
          # to test the validation of delete same actions in performance_lib
          # if intege_seq[i] <= intege_seq[i-1]:
          #     print(performance_sequence[i-3:i+5])
          #     print(intege_seq[i-3:i+5])
          assert intege_seq[i] > intege_seq[i-1], "pitch order, %d" % i
    # Libo---------------------revise check order and shift--------------------------------------------------

    # for i, event in enumerate(performance_sequence):
    #   if event.event_type == PerformanceEvent.TIME_SHIFT:
    #     shift_index.append(i)
    # import numpy as np
    # cc = np.array(intege_seq)
    # a=(cc > end_pitch)

  # def note_order_performance(self, performance):
  #     performance_0 = performance_lib.Performance(
  #         steps_per_second=performance_lib.DEFAULT_STEPS_PER_SECOND,
  #         start_step=0,
  #         num_velocity_bins=self._num_velocity_bins)
  #     dup_performance = copy.deepcopy(performance)
  #     leng = len(performance._events)
  #     for i in range(leng):
  #         if performance._events[i].event_type == 1:
  #             i = i + 1
  #     # return performance
# Libo---------------------useless--------------------------------------------------

def get_pipeline(config, min_events, max_events, eval_ratio):
  """Returns the Pipeline instance which creates the RNN dataset.

  Args:
    config: A PerformanceRnnConfig.
    min_events: Minimum number of events for an extracted sequence.
    max_events: Maximum number of events for an extracted sequence.
    eval_ratio: Fraction of input to set aside for evaluation set.

  Returns:
    A pipeline.Pipeline instance.
  """
  # Stretch by -5%, -2.5%, 0%, 2.5%, and 5%.
  stretch_factors = [0.95, 0.975, 1.0, 1.025, 1.05]

  # Transpose no more than a major third.
  transposition_range = range(-3, 4)
  # Libo---------Transposition------------important information-------------------------

  partitioner = pipelines_common.RandomPartition(
      music_pb2.NoteSequence,
      ['eval_performances', 'training_performances'],
      [eval_ratio])
  dag = {partitioner: dag_pipeline.DagInput(music_pb2.NoteSequence)}

  for mode in ['eval', 'training']:
    sustain_pipeline = note_sequence_pipelines.SustainPipeline(
        name='SustainPipeline_' + mode)
    stretch_pipeline = note_sequence_pipelines.StretchPipeline(
        stretch_factors, name='StretchPipeline_' + mode)
    splitter = note_sequence_pipelines.Splitter(
        hop_size_seconds=30.0, name='Splitter_' + mode)
    quantizer = note_sequence_pipelines.Quantizer(
        steps_per_second=config.steps_per_second, name='Quantizer_' + mode)
    transposition_pipeline = note_sequence_pipelines.TranspositionPipeline(
        transposition_range, name='TranspositionPipeline_' + mode)
    perf_extractor = PerformanceExtractor(
        min_events=min_events, max_events=max_events,
        num_velocity_bins=config.num_velocity_bins,
        name='PerformanceExtractor_' + mode)
    encoder_pipeline = encoder_decoder.EncoderPipeline(
        performance_lib.Performance, config.encoder_decoder,
        name='EncoderPipeline_' + mode)

    dag[sustain_pipeline] = partitioner[mode + '_performances']
    dag[stretch_pipeline] = sustain_pipeline
    dag[splitter] = stretch_pipeline
    dag[quantizer] = splitter
    dag[transposition_pipeline] = quantizer
    dag[perf_extractor] = transposition_pipeline
    dag[encoder_pipeline] = perf_extractor
    dag[dag_pipeline.DagOutput(mode + '_performances')] = encoder_pipeline

  return dag_pipeline.DAGPipeline(dag)


def main(unused_argv):
  tf.logging.set_verbosity(FLAGS.log)

  pipeline_instance = get_pipeline(
      min_events=32,
      max_events=512,
      eval_ratio=FLAGS.eval_ratio,
      config=performance_model.default_configs[FLAGS.config])

  input_dir = os.path.expanduser(FLAGS.input)
  output_dir = os.path.expanduser(FLAGS.output_dir)
  tttest = pipeline.tf_record_iterator(input_dir, pipeline_instance.input_type)
  # print(tttest)
  # sequences = []
  # for tt in tttest:
  #     sequences.append(tt)
  # print(tttest)
  # for tt in tttest:
  #     print("okay")
  # print(tttest)
  # ------------------libo: why it's wrong-------------------------
  pipeline.run_pipeline_serial(
      pipeline_instance,
      tttest,
      output_dir)


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
