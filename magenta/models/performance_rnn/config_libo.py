"""
run some test function

"""
import pickle
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

# generated_sequence.notes

# performance_sequence_generator.py
# This model can leave hanging notes. To avoid cacophony we turn off any note
# after 5 seconds (500 steps).
# MAX_NOTE_DURATION_SECONDS = 5.0

# Ensure that the track extends up to the step we want to start generating.
# performance.set_length(generate_start_step - performance.start_step)

NUM_OUTPUTS = 15
CONFIG = 'performance'
BUNDLE_PATH = "/home/zha231/Downloads/performance.mag"
NUM_STEPS = 200
PRIMER_MELODY = "[60, -2, 60, -2, 67, -2, 67, -2]"
# PRIMER_MELODY = "[]"
PRIMER_PITCHES = ''
PRIMER_MIDI = ''

# def para_libo():
#     NUM_OUTPUTS = 1
#     CONFIG = 'performance'
#     BUNDLE_PATH = "/home/zha231/Downloads/performance.mag"
#     NUM_STEPS = 3000
#     PRIMER_MELODY = "[60, -2, 60, -2, 67, -2, 67, -2]"
#     PRIMER_PITCHES = ''
#     PRIMER_MIDI = ''
#     return NUM_OUTPUTS, CONFIG, BUNDLE_PATH, NUM_STEPS, PRIMER_MELODY

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'run_dir', None,
    'Path to the directory where the latest checkpoint will be loaded from.')
tf.app.flags.DEFINE_string(
    'bundle_file', BUNDLE_PATH,
    'Path to the bundle file. If specified, this will take priority over '
    'run_dir, unless save_generator_bundle is True, in which case both this '
    'flag and run_dir are required')
tf.app.flags.DEFINE_boolean(
    'save_generator_bundle', False,
    'If true, instead of generating a sequence, will save this generator as a '
    'bundle file in the location specified by the bundle_file flag')
tf.app.flags.DEFINE_string(
    'bundle_description', None,
    'A short, human-readable text description of the bundle (e.g., training '
    'data, hyper parameters, etc.).')
tf.app.flags.DEFINE_string(
    'config', CONFIG, 'Config to use.')
tf.app.flags.DEFINE_string(
    'output_dir', '/tmp/performance_rnn/generated',
    'The directory where MIDI files will be saved to.')
tf.app.flags.DEFINE_integer(
    'num_outputs', NUM_OUTPUTS,
    'The number of tracks to generate. One MIDI file will be created for '
    'each.')
tf.app.flags.DEFINE_integer(
    'num_steps', NUM_STEPS,
    'The total number of steps the generated track should be, priming '
    'track length + generated steps. Each step is 10 milliseconds.')
tf.app.flags.DEFINE_string(
    'primer_pitches', PRIMER_PITCHES,
    'A string representation of a Python list of pitches that will be used as '
    'a starting chord with a quarter note duration. For example: '
    '"[60, 64, 67]"')
tf.app.flags.DEFINE_string(
    'primer_melody', PRIMER_MELODY,
    'A string representation of a Python list of '
    'magenta.music.Melody event values. For example: '
    '"[60, -2, 60, -2, 67, -2, 67, -2]". The primer melody will be played at '
    'a fixed tempo of 120 QPM with 4 steps per quarter note.')
tf.app.flags.DEFINE_string(
    'primer_midi', PRIMER_MIDI,
    'The path to a MIDI file containing a polyphonic track that will be used '
    'as a priming track.')
tf.app.flags.DEFINE_float(
    'temperature', 1.0,
    'The randomness of the generated tracks. 1.0 uses the unaltered '
    'softmax probabilities, greater than 1.0 makes tracks more random, less '
    'than 1.0 makes tracks less random.')
tf.app.flags.DEFINE_integer(
    'beam_size', 1,
    'The beam size to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'branch_factor', 1,
    'The branch factor to use for beam search when generating tracks.')
tf.app.flags.DEFINE_integer(
    'steps_per_iteration', 1,
    'The number of steps to take per beam search iteration.')
tf.app.flags.DEFINE_string(
    'log', 'INFO',
    'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
    'or FATAL.')
tf.app.flags.DEFINE_string(
    'hparams', '',
    'Comma-separated list of `name=value` pairs. For each pair, the value of '
    'the hyperparameter named `name` is set to `value`. This mapping is merged '
    'with the default hyperparameters.')




