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
from scipy.special import logsumexp

# def notesequence_from_dir(input_dir, output_file=None, num_threads=1, recursive=False):
#     """Converts files to NoteSequences and writes to `output_file`.
#     if there are too many midi files, call the convert_directory in scripts
#     ------------------------write that in algorithm------------------------
#     Input files found in `root_dir` are converted to NoteSequence protos with the
#     basename of `root_dir` as the collection_name, and the relative path to the
#     file from `root_dir` as the filename. If `recursive` is true, recursively
#     converts any subdirectories of the specified directory.
#
#     Args:
#       input_dir: A string specifying a root directory.
#       output_file: Path to TFRecord file to write results to.
#       num_threads: The number of threads to use for conversions.
#                     'Number of worker threads to run in parallel.'
#       recursive: A boolean specifying whether or not recursively convert files
#           contained in subdirectories of the specified directory.
#     :return: a list of note sequences
#     """
#     # from magenta.scripts.convert_dir_to_note_sequences import convert_directory
#
#     if not input_dir:
#         tf.logging.fatal('--input_dir required')
#         return
#     root_dir = os.path.expanduser(input_dir)
#
#     # convert_directory(input_dir, output_file, num_threads, recursive)
#     sequences = []
#     with futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
#         future_to_path = queue_conversions(root_dir, '', pool, recursive)
#         for future in futures.as_completed(future_to_path):
#             path = future_to_path[future]
#             try:
#                 sequence = future.result()
#             except Exception as exc:  # pylint: disable=broad-except
#                 tf.logging.fatal('%r generated an exception: %s', path, exc)
#
#             if sequence:
#                 sequences.append(sequence)
#
#     if output_file:
#         output_file = os.path.expanduser(output_file)
#         output_dir = os.path.dirname(output_file)
#         tf.gfile.MakeDirs(output_dir)
#         with note_sequence_io.NoteSequenceRecordWriter(output_file) as writer:
#             sequences_written = 0
#             for sequence in sequences:
#                 writer.write(sequence)
#                 sequences_written += 1
#             tf.logging.log_every_n(
#                 tf.logging.INFO, "Wrote %d of %d NoteSequence protos to '%s'", 100,
#                 sequences_written, len(future_to_path), output_file)
#
#         tf.logging.info("Wrote %d NoteSequence protos to '%s'", sequences_written,
#                         output_file)
#     else:
#         tf.logging.info("return note sequences without writing into file")
#
#     return sequences


def load_from_notesequence_file(input_file):
    """
    :param input_file: 'TFRecord to read NoteSequence protos from.'
    :param input_type: NoteSequence
    :return: a list of note sequences
    """
    input_type = NoteSequence
    input_path = os.path.expanduser(input_file)
    tf_record_iterator = pipeline.tf_record_iterator(input_path, input_type)
    sequences = []
    for sequence in tf_record_iterator:
      sequences.append(sequence)
    return sequences


def notesequences_from_sequences(primer_pitches=None, primer_melody=None, primer_midi_path=None):
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
     :param primer_midi_path: The path to a MIDI file containing a polyphonic track
    :return: a list of note sequences
    # def spam(a, b=None, c=None):
    #     print(b)
    #     print(c)
    # spam(100, c=0)
    # spam(100)
    """
    sequences = []
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
        sequences.append(primer_sequence)

    if primer_melody:
        primer_melody = magenta.music.Melody(ast.literal_eval(primer_melody))
        primer_sequence = primer_melody.to_sequence()
        sequences.append(primer_sequence)
        # melodies_lib: Converts the Melody to NoteSequence proto.

    if primer_midi_path:
        primer_midi_path = os.path.expanduser(primer_midi_path)
        primer_sequence = magenta.music.midi_file_to_sequence_proto(primer_midi_path)
        sequences.append(primer_sequence)

    if len(sequences) < 1:
        tf.logging.warning(
            'No priming sequence specified. Defaulting to empty sequence.')
    # primer_sequence = music_pb2.NoteSequence()
    # primer_sequence.ticks_per_quarter = constants.STANDARD_PPQ

    return sequences


def split_Notesequences(note_sequences, hop_size_seconds=None, check_last_len=False,
                        cut_points_seconds=None, start_time_shift=False):
    """Split NoteSequences into many using a fixed hop size or other input points.
    !!!!Note: Each of the resulting NoteSequences is shifted to start at time zero.

    The NoteSequence will be split at all hop positions, regardless of whether or not any
    notes are sustained across the potential split time, thus sustained notes will be truncated.

    This function splits a NoteSequence into multiple NoteSequences, all of fixed
    size (unless `split_notes` is False, in which case splits that would have
    truncated notes will be skipped; i.e. each split will either happen at a
    multiple of `hop_size_seconds` or not at all).

      Args:
        note_sequence: The NoteSequence to split.

        hop_size_seconds: The hop size, in seconds, at which the NoteSequence will
            be split.
        check_last_len: The last section which is not long enough will be abandoned.

        cut_points_seconds: measured by seconds, the start and ending time is not included

        skip_splits_inside_notes(delete)

      Returns:
        A Python list of list of NoteSequences.
      """

    assert (hop_size_seconds or cut_points_seconds)
    all_subsequences = []

    for note_sequence in note_sequences:
        prev_split_time = 0.0
        subsequences = []
        time_cut = []
        last_sect_save = True

        if hop_size_seconds and note_sequence.total_time > hop_size_seconds:
            time_cut = np.arange(hop_size_seconds, note_sequence.total_time, hop_size_seconds)
            if check_last_len and (hop_size_seconds > note_sequence.total_time - time_cut[-1]):
                last_sect_save = False
        elif cut_points_seconds and note_sequence.total_time > cut_points_seconds[-1]:
            time_cut = cut_points_seconds
        # The reason why the if clause is not at the start of the function is,
        # different elements (note sequence) may don't have the same time periods.

        for split_time in time_cut:
            subsequence = extract_subsequence(note_sequence, prev_split_time, split_time)
            # the function is in sequence_lib
            if start_time_shift and prev_split_time > 0:
                subsequence = note_sequence_shift_time(subsequence, prev_split_time)
                # or subsequence.notes[0].start_time
            subsequences.append(subsequence)
            prev_split_time = split_time

        # note_sequence.total_time > prev_split_time is always true, because of np.arrange
        if note_sequence.total_time > prev_split_time and last_sect_save:
            subsequence = extract_subsequence(note_sequence, prev_split_time,
                                              note_sequence.total_time)
            if start_time_shift and prev_split_time > 0:
                subsequence = note_sequence_shift_time(subsequence, prev_split_time)
            subsequences.append(subsequence)

        all_subsequences.append(subsequences)

    return all_subsequences

def note_sequence_shift_time(subsequence, time_shift):
    """
    shift a note sequence to another time period
    !!!Note: After time shift, subsequence.total_time is only the end time of
            the sequence, which was the time duration when the start time is zero.
    :param subsequence: NoteSequence
    :param time_shift: shift time
    :return:
    """

    # quantized_sequence.notes.pop(-1)
    # del quantized_sequence.notes[-1:-10]
    # quantized_sequence.total_time
    # the total_time need to be updated after delete or added
    # quantized_sequence.notes[-1].end_time
    assert isinstance(subsequence, NoteSequence), "not a NoteSequence"
    assert time_shift != 0
    if time_shift == 0:
        return subsequence
    elif time_shift + subsequence.notes[0].start_time < 0:
        print("time shift + start time < 0, return original sequence")
        return subsequence
    for i in range(len(subsequence.notes)):
        subsequence.notes[i].start_time = subsequence.notes[i].start_time + time_shift
        subsequence.notes[i].end_time = subsequence.notes[i].end_time + time_shift

    subsequence.total_time = subsequence.total_time + time_shift
    # total_time: ending time
    # subsequence.subsequence_info.start_time_offset = time_shift + subsequence.notes[0].start_time
    return subsequence


def performances_from_Notesequences(primer_sequences,
                                    min_events_discard=1, max_events_truncate=None, no_start_shift=False):
    """
    Extracts performances from the given non-empty NoteSequence.
    situation 1: the real start time of sequence is start_time_offset, but in the note sequence
                  the start time is set as 0. (original magenta)
    situation 2: the real start time of sequence is start_time_offset, the note start at real time
    :param primer_sequences: a list of Note sequences
    :param start_step: Start extracting a sequence at this time step.------------delete----------------
    :param min_events_discard: Minimum length of tracks in events. Shorter tracks are discarded.
    :param max_events_truncate: Maximum length of tracks in events. Longer tracks are truncated.

    :return:performance_lib.Performance
    """
    assert primer_sequences != []
    steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
    num_velocity_bins = 0
    performances = []
    # start_step = 0
    for sequence in primer_sequences:
        quantized_sequence = mm.quantize_note_sequence_absolute(sequence, steps_per_second)
        # start_time: 9.99595359375
        # end_time: 10.0
        # quantized_start_step: 1000
        # quantized_end_step: 1001
        # small problem here--------------------------------------------------------------------------
        start_inf = quantized_sequence.subsequence_info.start_time_offset * steps_per_second
        time_first_action = quantized_sequence.notes[0].quantized_start_step
        if no_start_shift:
            start_step = time_first_action
            # start from the first event or action, situation 1 or 2, no first shift
        elif start_inf > time_first_action:
            start_step = 0
            # situation 1 (start_inf(2000) > time_first_action(6)) and allow shift
        else:
            start_step = start_inf
            # situation 2 (start_inf(2000) <= time_first_action(2006 or 2000)), allow shift.

        performance = performance_lib.Performance(quantized_sequence, start_step=start_step,
                                  num_velocity_bins=num_velocity_bins)
        if (max_events_truncate is not None and len(performance) > max_events_truncate):
            performance.truncate(max_events_truncate)
        if len(performance) < min_events_discard:
            print('delete one short sequence(%d-length)', len(performance))
        else:
            performances.append(performance)

    # might be empty if no input.
    return performances


def performance_from_Notesequence(primer_sequence,
                                  min_events_discard=None, max_events_truncate=None, no_start_shift=False):
    """
    Extracts performance from the given non-empty NoteSequence.
    :param primer_sequence: a list of Note sequences
    :param start_step: Start extracting a sequence at this time step.--------------delete-------------
    :param min_events_discard: Minimum length of tracks in events. Shorter tracks are discarded.
    :param max_events_truncate: Maximum length of tracks in events. Longer tracks are truncated.

    :return:performance_lib.Performance
    """
    steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
    num_velocity_bins = 0
    quantized_sequence = mm.quantize_note_sequence_absolute(primer_sequence, steps_per_second)
    start_inf = quantized_sequence.subsequence_info.start_time_offset * steps_per_second
    time_first_action = quantized_sequence.notes[0].quantized_start_step
    if no_start_shift:
        start_step = time_first_action
        # start from the first event or action, situation 1 or 2, no first shift
    elif start_inf > time_first_action:
        start_step = 0
        # situation 1 (start_inf(2000) > time_first_action(6)) and allow shift
    else:
        start_step = start_inf
        # situation 2 (start_inf(2000) <= time_first_action(2006 or 2000)), allow shift.
    performance = performance_lib.Performance(quantized_sequence, start_step=start_step,
                              num_velocity_bins=num_velocity_bins)

    if (max_events_truncate is not None and len(performance) > max_events_truncate):
        performance.truncate(max_events_truncate)

    if (min_events_discard is not None and len(performance) < min_events_discard):
        print('discard one short sequence(%d-length)', len(performance))
        return
    # start_step won't change with the pop() or delete, num_steps and end_step will change
    # start_step can't be set after performance is created.

    return performance

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


def weight_to_section(performance, generator,
                               end_time_steps, first_event_next_section, hotcoding, final_state=None):
    """
    freely sample to the next section. It is different from sample to next event.
    The structure of next section is fixed. After shift to the given time, jump
    (not sample) to the start event of the next section.

    :param generator: generate settings
    :param first_event_next_section: the start event of the next section
    :param end_time_steps: the specific end time step, also is the time step of start event
    :param args: generate options
    :return: weight
    two situations:
    1. no input state: generate (sample) from the first step, since the "inputs" is different
         from the following step, and the first step need to construct the initial state
    2. input state: generate (sample) with the input state
    """

    assert isinstance(first_event_next_section, PerformanceEvent), "not a Performance Event"
    assert end_time_steps > performance.num_steps, "time_step is not correct"
    # print('free sample to time_step %d' % end_time_steps)
    event_sequences = [performance]

    if final_state:
        inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
        final_state, softmax = generator._model.one_step_forward(inputs, final_state)
    else:
        softmax, final_state, _ = generator._model.first_step_forward(event_sequences)

    indices = generator._model._config.encoder_decoder.extend_event_sequences(
        event_sequences, softmax)

    while performance.end_step < end_time_steps:
        # state_prev = copy.deepcopy(final_state)  # unnecessary
        inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
        final_state, softmax = generator._model.one_step_forward(inputs, final_state)
        indices = generator._model._config.encoder_decoder.extend_event_sequences(
            event_sequences, softmax)
    assert performance._events[-1].event_type == 3
    # performance[-1].event_type: the same with the upper expression
    # last shift run across the time constraint
    # pull back the last shift (if the last shift choice is perfect, that's okay. The weight
    # (probability) is always the most important, not only the sequences or choices)
    performance._events.pop()
    time_step_gap = end_time_steps - performance.end_step
    shift_event = PerformanceEvent(event_type=3, event_value=np.int(time_step_gap))
    performance._events.append(shift_event)
    # performance.set_length(np.int(end_time_steps))
    # upper is wrong: the num_steps (duration) is the correct input argument, not end_step
    # pull back------------------------------------------------------------------------------

    index_right_shift = hotcoding.encode_event(shift_event)
    fd_shift = softmax[0][-1][index_right_shift]
    # three-dimensional:  particle, step, position
    Fd_last = sum(softmax[0][-1][0:index_right_shift])
    # it consist of two items in the table: the item before the last (shift) is an action
    # 1: possibility of pitches bigger than the pitch of last action
    #               less than that pitch is set impossible in our algorithm
    # 2: shift step is less than the last shift step
    # probability in weight about last step------------------------------------------------

    index_next = hotcoding.encode_event(first_event_next_section)
    # performance._events.append(first_event_next_section)
    inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
    # here a list of one hot encoding about last event
    final_state_next, softmax_next = generator._model.one_step_forward(inputs, final_state)
    # performance._events.pop()
    fd_next = softmax_next[0][-1][index_next]
    Fd_next = fd_shift * np.sum(softmax_next[0][-1][0:index_next+1])
    # the last item in F in the table (sum to the pitch of next event)
    # probability in weight about the next step----------------------------------------------------

    fd = fd_shift * fd_next
    Fd = Fd_last + Fd_next
    weight = fd / (1 - Fd)
    # print("f, %s; F, %s; weight, %s" % (fd, Fd, weight))
    return weight, final_state


def weight_to_next_action(performance, generator, next_event, state, hotcoding):
    """
    action in the given section.
    :param generator: generate options
    :param next_event: an instance of PerformanceEvent
    :param args: generate options
    :return: weight
    """
    # assert isinstance(performance_next, performance_lib.Performance), "not a Performance"
    event_sequences = [performance]
    inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
    final_state, softmax = generator._model.one_step_forward(inputs, state)
    performance._events.append(next_event)
    index_next = hotcoding.encode_event(next_event)
    weight = softmax[0][-1][index_next]

    return weight, final_state

# def weight_1generate_to_section(performance, generator,
#                                end_time_steps, first_event_next_section, args, final_state):
#     """
#     freely sample to the next section. It is different from sample to next event.
#     The structure of next section is fixed. After shift to the given time, jump
#     (not sample) to the start event of the next section.
#     Note: not correct for the first step
#
#     :param generator: generate settings
#     :param first_event_next_section: the start event of the next section
#     :param end_time_steps: the specific end time step, also is the time step of start event
#     :param args: generate options
#     :return: weight
#     """
#
#     assert isinstance(first_event_next_section, PerformanceEvent), "not a Performance Event"
#     assert end_time_steps > performance.num_steps, "time_step is not correct"
#     # print('free sample to time_step %d' % end_time_steps)
#     event_sequences = []
#     event_sequences.append(performance)
#     hotcoding = PerformanceOneHotEncoding()
#
#     while performance.end_step < end_time_steps:
#         # state_prev = copy.deepcopy(final_state)  # unnecessary
#         inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
#         final_state, softmax = generator._model.one_step_forward(inputs, final_state)
#         indices = generator._model._config.encoder_decoder.extend_event_sequences(
#             event_sequences, softmax)
#
#     assert performance._events[-1].event_type == 3
#     # performance[-1].event_type: the same with the upper expression
#     # last shift run across the time constraint
#     # pull back the last shift (if the last shift choice is perfect, that's okay. The weight
#     # (probability) is always the most important, not only the sequences or choices)
#     performance._events.pop()
#     time_step_gap = end_time_steps - performance.end_step
#     shift_event = PerformanceEvent(event_type=3, event_value=np.int(time_step_gap))
#     performance._events.append(shift_event)
#     # performance.set_length(np.int(end_time_steps))
#     # wrong use about the set_length: in the function the num_steps (duration) is the input argument, not end_step
#     # pull back------------------------------------------------------------------------------
#
#     index_right_shift = hotcoding.encode_event(shift_event)
#     fd_shift = softmax[0][-1][index_right_shift]
#     # three-dimensional:  particle, step, position
#     Fd_last = sum(softmax[0][-1][0:index_right_shift])
#     # it consist of two items in the table: the item before the last (shift) is an action
#     # 1: possibility of pitches bigger than the pitch of last action
#     #               less than that pitch is set impossible in our algorithm
#     # 2: shift step is less than the last shift step
#     # probability in weight about last step------------------------------------------------
#
#     index_next = hotcoding.encode_event(first_event_next_section)
#     performance._events.append(first_event_next_section)
#     inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
#     # here a list of one hot encoding about last event
#     final_state_next, softmax_next = generator._model.one_step_forward(inputs, final_state)
#     performance._events.pop()
#     fd_next = softmax_next[0][-1][index_next]
#     Fd_next = fd_shift * sum(softmax_next[0][-1][0:index_next+1])
#     # the last item in F in the table (sum to the pitch of next event)
#     # probability in weight about the next step----------------------------------------------------
#
#     fd_log = np.log(fd_shift) + np.log(fd_next)
#     Fd_log = np.log(1-(Fd_last + Fd_next))
#     weight_log = fd_log - Fd_log
#     print("f, %d; F, %d; weight, %d" % (fd_log, Fd_log, weight_log))
#     return weight_log, final_state



# def weight_jump_to_individual_action(performance, generator, next_event, end_time_steps, args, state):
#     """
#     jump to the a non-shift given event
#     :param generator: generate options
#     :param next_event: an instance of PerformanceEvent
#     :param args: generate options
#     :return: weight
#     """
#     assert isinstance(next_event, PerformanceEvent), "not a Performance Event"
#     time_step_gap = end_time_steps - performance.end_step
#     assert next_event.event_type != 3, "time gap is wrong"
#     assert 0 <= time_step_gap <= performance_lib.MAX_SHIFT_STEPS, "time gap is wrong"
#     event_sequences = []
#     event_sequences.append(performance)
#
#     hotcoding = PerformanceOneHotEncoding()
#     index_next_event = hotcoding.encode_event(next_event)
#     inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
#     final_state, softmax = generator._model.one_step_forward(inputs, state)
#
#     if time_step_gap == 0:
#         assert index_next_event > hotcoding.encode_event(performance._events[-1]), "same time event order wrong"
#         performance._events.append(next_event)
#         weight_log = np.log(softmax[0][-1][index_next_event])
#     else:
#         shift_event = PerformanceEvent(event_type=3, event_value=np.int(time_step_gap))
#         index_shift = hotcoding.encode_event(shift_event)
#         performance._events.append(shift_event)
#         w_shift_log = np.log(softmax[0][-1][index_shift])
#         final_state, softmax = generator._model.one_step_forward(inputs, final_state)
#         performance._events.append(next_event)
#         w_event_log = np.log(softmax[0][-1][index_next_event])
#         # weight_log = w_shift * w_event
#         weight_log = w_shift_log + w_event_log
#
#     return weight_log


class sampler_given_sections(object):
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
        self.steps_per_second = performance_lib.DEFAULT_STEPS_PER_SECOND
        self.args = args
        self.num_particles = int(num_particles)
        self.num_velocity_bins = num_velocity_bins

    def generate_to_fix_section(self, performance_prev, performance_next, hotcoding):
        """
        genrate different particles(samples) with the same input or different inputs
        !!!Note: if the start action of next performance must be turn on or turn off,
                 it may cause trouble since the last action in last section may be shift

        !!!!The measure: if the first action is a shift, give the time to the generation

        :param performance_prev:  performance
        :param performance_next: start from real start time, not 0;
        :return:
        """
        assert performance_next.start_step > performance_prev.end_step
        # if performance_next[0].event_type != 3:
        #     arrive_step = performance_next.start_step
        #     index_start = 0
        # else:
        #     arrive_step = performance_next.start_step + performance_next[0].event_value
        #     index_start = 1
        # # arrive_step is not correct, the sample may shift several times in the .
        assert performance_next[0].event_type == 3
        index_start = 0
        arrive_step = performance_next.start_step
        next_start_event = performance_next._events[index_start]

        self.perf_list = []
        self.w = []
        self.final_state = []
        # to encode and decode event
        for i in range(self.num_particles):
            if isinstance(performance_prev, performance_lib.Performance):
                performance_i = copy.deepcopy(performance_prev)
            else:
                performance_i = performance_prev[i]
            weight_section, state_i = weight_to_section(performance_i, self.generator,
                                                        arrive_step, next_start_event, hotcoding)
            self.final_state.append(state_i)
            self.w.append(weight_section)
            self.perf_list.append(performance_i)

        re_index = self.systematic_resample()
        self.perf_list = [copy.deepcopy(self.perf_list[i]) for i in re_index]
        self.final_state = [copy.deepcopy(self.final_state[i]) for i in re_index]

        num_copy = 0
        for event in performance_next._events[index_start:]:
            for i in range(self.num_particles):
                weight_action, self.final_state[i] = weight_to_next_action(self.perf_list[i],
                                                                     self.generator, event, self.final_state[i], hotcoding)
                self.w[i] = weight_action

            re_index = self.systematic_resample()
            # there are many steps in performance_next, in which most of the index is the same with the original
            if not np.array_equal(re_index, range(self.num_particles)):
                num_copy += 1
                print(self.perf_list[0].end_step)
                self.perf_list = [copy.deepcopy(self.perf_list[i]) for i in re_index]
                self.final_state = [copy.deepcopy(self.final_state[i]) for i in re_index]
                # to reduce the copy times.
        print("number of copy, %s; num of actions in the next section, %s" % (num_copy, len(performance_next)))
        return self.perf_list


    def systematic_resample(self):
        """
        resampling based on the weights: simply duplicate and delete particles
        param: w, weights or log likely hood of weights, "list"
        return: a, the select index of particles, start from 0
        """
        w = np.array(self.w)
        # ndarray, not 1*n array
        # w = np.exp(w_log)
        # weight are log likely hood, converted it into normal format
        w = w / np.sum(w)
        # w_norm_log = w_log - logsumexp(w_log)
        # w = np.exp(w_norm_log)
        # print(w)
        # print(sum(w))
        n = len(w)
        assert n > 1
        u = np.random.rand() / n
        s = w[0]
        j = 0
        re_index = np.zeros(n, dtype=int)
        ninv = float(1) / n
        # or 1.0/n , different form python 3,
        for k in range(n):
            while s < u:
                j += 1
                s += w[j]
            re_index[k] = j
            u += ninv
        return re_index


def performance_to_notes_to_music(num_outputs, perf_list,
                                  output_dir, max_note_duration, end_time=None):
    """
    Save the first num_outputs sequence in perf_list.
    give weight a small value if the duration is more than 5 seconds.!!!!!!!!!!!!!!!!!
    if some notes don't have corresponding turn off. !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    :param output_dir: output directory
    :param max_note_duration
    :param end_time: specificed time
    :return:
    """
    # assert len(perf_list) == int(num_outputs)
    if not output_dir:
        tf.logging.fatal('--output_dir required')
        return
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)

    date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
    digits = len(str(num_outputs))

    generated_note_sequences = []
    for i in range(num_outputs):
        performance = copy.deepcopy(perf_list[i])
        # actually unnecessary, since the changes in writing process is good to reflect
        generated_sequence = performance.to_sequence(max_note_duration=max_note_duration)
        if end_time:
            assert (generated_sequence.total_time - end_time) <= 1e-5
        midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
        midi_path = os.path.join(output_dir, midi_filename)
        magenta.music.sequence_proto_to_midi_file(generated_sequence, midi_path)
        generated_note_sequences.append(generated_sequence)

    print('Wrote %d MIDI files to %s' % (num_outputs, output_dir))
    # tf.logging.info('Wrote %d MIDI files to %s' % (1, output_dir))
    return generated_note_sequences


def loglike_from_note_sequences(generator, generated_note_sequences):
    """
    The note sequence is firstly converted into performance, then get the new
    :param generator:
    :param generated_note_sequences:
    :param args:
    :return:
    """
    softmaxes = []
    performances = []
    for note_sequence in generated_note_sequences:
        # start_step = note_sequence.notes[0].start_time
        performance = performance_from_Notesequence(note_sequence)
        event_sequences = [performance]
        softmax, final_state, _ = generator._model.first_step_forward(event_sequences)
        performances.append(performance)
        softmaxes.append(softmax[0, :-1, :])
        # that is a sudo forward, we want to get all the probability about the current sequence
        # ":-1" delete the last softmax, which is about the next step.
    all_loglik = evaluate_log_likelihood(performances, softmaxes)
    return all_loglik


def evaluate_log_likelihood(event_sequences, softmaxes):
    """Evaluate the log likelihood of multiple event sequences.

    Each event sequence is evaluated from the end. If the size of the
    corresponding softmax vector is 1 less than the number of events, the entire
    event sequence will be evaluated (other than the first event, whose
    distribution is not modeled). If the softmax vector is shorter than this,
    only the events at the end of the sequence will be evaluated.

    Args:
      event_sequences: A list of EventSequence objects.
      softmaxes: A list of (list of softmax) probability vectors. The list of
          softmaxes should be the same length as the list of event sequences.
          [i, j, k] the first one means the i-th performance sequence,
          the second one means the j-th event in i-th sequence
          the third is for the probability of k-th position in j-th event of i-th sequence

    Returns:
      A Python list containing the log likelihood of each event sequence.

    Raises:
      ValueError: If one of the event sequences is too long with respect to the
          corresponding softmax vectors.
    """
    all_loglik = []
    for i in xrange(len(event_sequences)):
        if len(softmaxes[i]) >= len(event_sequences[i]):
            raise ValueError(
                'event sequence must be longer than softmax vector (%d events but '
                'softmax vector has length %d)' % (len(event_sequences[i]),
                                                   len(softmaxes[i])))
        end_pos = len(event_sequences[i])
        start_pos = end_pos - len(softmaxes[i])
        loglik = []
        # loglik = 0
        for softmax_step, position in enumerate(range(start_pos, end_pos)):
            hotcoding = PerformanceOneHotEncoding()
            index_next = hotcoding.encode_event(event_sequences[i][position])
            #-------------libo----------------------sequence, event------------------------
            loglik_pos = np.log(softmaxes[i][softmax_step][index_next])
            # ----------libo-------sequence,  event, index_next---------------------------------
            # loglik += np.log(softmaxes[i][softmax_step][index_next])
            loglik.append(loglik_pos)
            # --------------libo a list of log likelihood in i-th sequence------------------
        all_loglik.append(loglik)

    return all_loglik


def prepare_data(section_seconds, num_selected_sections, cut_points):
    output_file = "~/data/notesequences11.tfrecord"
    # input_dir = "~/data/data_2011"
    # notesequences = notesequence_from_dir(input_dir, output_file)
    notesequences = load_from_notesequence_file(output_file)
    # obtain the note sequences

    note_sections = split_Notesequences(notesequences,
                                        hop_size_seconds=section_seconds, check_last_len=True)
    # obtain small sections of all the note sequences

    note_sections_selected = []
    # a note sequence list
    for i in range(num_selected_sections):
    # for note_section in note_sections:
        note_sections_selected.append(note_sections[i][1])
        # obtain selected sections in each original note sequence
    # a better way is to extract the necessary time period of sequence-----------------------------------
    # # test------------------------
    # config = FLAGS.config
    # bundle_file = FLAGS.bundle_file
    # aaa = generator_bundle_args(bundle_file=bundle_file)
    # generator = aaa.get_generator(config_name=config)
    # generator.initialize()
    # hhh = loglike_from_note_sequences(generator, note_sections_selected)
    # # test------------------------

    note_subsections = split_Notesequences(note_sections_selected,
                                           cut_points_seconds=cut_points, start_time_shift=True)
    # a list of list of subsections, which will be adopted to generate and test

    performance_subsections = []
    for note_subsection in note_subsections:
        performance_subsection = performances_from_Notesequences(note_subsection, no_start_shift=True)
        performance_subsections.append(performance_subsection)
    # obtain list of list of performances
    return performance_subsections

def brutal_sampler_section(generator, performance_prev, performance_next, final_state=None):
    """

    :param num_particles:
    :param generator:
    :param performance_prev:
    :param performance_next:
    :param final_state:
    :return:
    """

    assert performance_next.start_step > performance_prev.end_step
    if performance_next[0].event_type != 3:
        end_time_steps = performance_next.start_step
        index_start = 0
    else:
        end_time_steps = performance_next.start_step + performance_next[0].event_value
        index_start = 1

    performance = copy.deepcopy(performance_prev)
    event_sequences = [performance]
    if final_state:
        inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
        final_state, softmax = generator._model.one_step_forward(inputs, final_state)
    else:
        softmax, final_state, _ = generator._model.first_step_forward(event_sequences)
    indices = generator._model._config.encoder_decoder.extend_event_sequences(
        event_sequences, softmax)

    while performance.end_step < end_time_steps:
        # state_prev = copy.deepcopy(final_state)  # unnecessary
        inputs = generator._model._config.encoder_decoder.get_inputs_batch(event_sequences)
        final_state, softmax = generator._model.one_step_forward(inputs, final_state)
        indices = generator._model._config.encoder_decoder.extend_event_sequences(
            event_sequences, softmax)

    assert performance._events[-1].event_type == 3
    # performance[-1].event_type: the same with the upper expression
    # last shift run across the time constraint
    # pull back the last shift (if the last shift choice is perfect, that's okay. The weight
    # (probability) is always the most important, not only the sequences or choices)
    performance._events.pop()
    time_step_gap = end_time_steps - performance.end_step
    shift_event = PerformanceEvent(event_type=3, event_value=np.int(time_step_gap))
    performance._events.append(shift_event)
    performance._events = performance._events + performance_next._events[index_start:]
    return event_sequences


def main(unused_argv):
    """
    now couldn't generate several sections in one sequence,one problem final state
    :param unused_argv:
    :return:
    """
    section_seconds = 30
    num_selected_sections = 1
    cut_points = [10, 20]
    # cut_points = [10, 20, 30, 40]------------------------------------------
    output_dir_smc = os.path.expanduser(FLAGS.output_dir)
    output_dir_brutal = os.path.expanduser(FLAGS.output_dir_brutal)
    output_dir_org = os.path.expanduser(FLAGS.output_dir_org)
    performance_subsections = prepare_data(section_seconds, num_selected_sections, cut_points)

    config = FLAGS.config
    bundle_file = FLAGS.bundle_file
    aaa = generator_bundle_args(bundle_file=bundle_file)
    generator = aaa.get_generator(config_name=config)
    max_note_duration = performance_sequence_generator.MAX_NOTE_DURATION_SECONDS
    num_particles = FLAGS.num_outputs
    args = {
        'temperature': FLAGS.temperature,
        'beam_size': FLAGS.beam_size,
        'branch_factor': FLAGS.branch_factor,
        'steps_per_iteration': FLAGS.steps_per_iteration
    }
    # may be not necessary----------------------
    generator.initialize()
    hotcoding = PerformanceOneHotEncoding()
    generation_pfrnn = sampler_given_sections(generator, num_particles, args)
    len_actions_all = []
    note_samples_org = []
    logs_org = []
    note_samples_brutal = []
    logs_brtual = []
    notes_sample_smc = []
    logs_smc = []


    for performance_subsection in performance_subsections:
        num_samples_per_generation = 1
        num_sections = len(performance_subsection)
        assert num_sections > 2 and num_sections % 2 == 1
        # int(num_sections) % 2
        # guarantee always generate a middle section
        len_actions = []
        performance_org = copy.deepcopy(performance_subsection[0])
        len_actions.append(len(performance_subsection[0]))
        for i in range(num_sections)[1:]:
            performance_org._events = performance_org._events + performance_subsection[i]._events
            len_actions.append(len(performance_subsection[i]))
        len_actions_all.append(len_actions)
        note_org = performance_to_notes_to_music(num_samples_per_generation, [performance_org],
                                                 output_dir_org, max_note_duration)
        log_org = loglike_from_note_sequences(generator, note_org)
        note_samples_org.append(note_org)
        logs_org.append(log_org)

        performance_prev = copy.deepcopy(performance_subsection[0])
        # performance_next = performance_subsection[2]
        # range(0, num_sections, 2)[:-1]
        # i,j in enumerate(range(2, num_sections, 2)):
        for i in range(2, num_sections, 2):
            performance_next = performance_subsection[i]
            performances_pfrnn = generation_pfrnn.generate_to_fix_section(performance_prev,
                                                                          performance_next, hotcoding)
            performance_prev = copy.deepcopy(performances_pfrnn)
            # the first generation is different from the following steps, since it inputs list of performance
            # multiple sections generation hasn't been tested-----one problem final state????------------------------
        note_samples_smc = performance_to_notes_to_music(num_samples_per_generation,
                                                        performances_pfrnn, output_dir_smc, max_note_duration)
        # the reason why compute the log likely hood after note sequences are generated:
        # when the sequence is converted to note sequence, un reasonable actions, like the no-on note off action
        # (as magenta declare that always happen) will be deleted. These will affect the sequence and the possibility.
        #  or "to sequence" and other functions
        log_smc = loglike_from_note_sequences(generator, note_samples_smc)
        notes_sample_smc.append(note_samples_smc)
        logs_smc.append(log_smc)

        section_formmer_brutal = copy.deepcopy(performance_subsection[0])
        for i in range(2, num_sections, 2):
            section_next = performance_subsection[i]
            performances_brutal = brutal_sampler_section(generator, section_formmer_brutal, section_next)
            section_formmer_brutal = copy.deepcopy(performances_brutal)
            # deep copy necessary?--------------------------------------
        note_brutle = performance_to_notes_to_music(num_samples_per_generation,
                                                     performances_brutal, output_dir_brutal, max_note_duration)
        log_brutal = loglike_from_note_sequences(generator, note_brutle)
        note_samples_brutal.append(note_brutle)
        logs_brtual.append(log_brutal)

def console_entry_point():
    tf.app.run(main)


if __name__ == '__main__':
    console_entry_point()