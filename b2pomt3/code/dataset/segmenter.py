import numpy as np
from utils.signal_stats import detect_R_peak
import random


class HeartBeatSegmenter(object):
    """Split ecg signal into individuals heartbeats.

    Args:
        segment_size(tuple or int): Desired segment size.
        take_average(boolean): average the heartbeats pairs
    """
    def __init__(self, segment_size=110, take_average=False):
        self.segment_size = segment_size
        self.take_average = take_average

    def __call__(self, sample):
        # find out the R peaks
        r_peak = detect_R_peak(sample)

        sample = sample.cpu().numpy()
        all_segments = []
        half_size_int = int(self.segment_size//2)

        for recording in range(len(sample)):
            segments = []
            # throw away first and last R peaks as it may lead to segments
            # of len less than desired length.
            for i in r_peak[recording][1: -1]:
                # make a segment
                new_heart_beat = sample[recording][int(
                    i)-int(half_size_int*0.8): int(i)+int(half_size_int*1.2)]
                if len(new_heart_beat) != self.segment_size:
                    continue
                # append it to the list
                segments.append(new_heart_beat)

            if self.take_average is True:
                segments = [np.mean(segments, axis=0)]

            all_segments.extend(segments)

        return np.array(all_segments)


class PairedHeartBeatSegmenter(object):
    """Split ecg signal into pairs of heartbeats.

    Args:
        segment_size(tuple or int): Desired segment size.
        take_average(boolean): average the heartbeats pairs
    """
    def __init__(self, segment_size=230, take_average=False):
        self.segment_size = segment_size
        self.take_average = take_average

    def __call__(self, sample):
        # find out the R peaks
        r_peak = detect_R_peak(sample)

        sample = sample.cpu().numpy()
        all_segments = []

        for recording in range(len(sample)):
            segments = []
            # throw away first and last R peaks as it may lead to segments
            # of len less than desired length.
            for i in range(1, len(r_peak[recording])-1):
                curr_peak = r_peak[recording][i]
                next_peak = r_peak[recording][i+1]
                pad_size = self.segment_size - (next_peak-curr_peak)
                # make a segment
                pair_begining = int(curr_peak)-int(np.round(pad_size*0.4))
                pair_end = int(next_peak)+int(np.round(pad_size*0.6))
                new_heart_beat = sample[recording][pair_begining:pair_end]

                if len(new_heart_beat) != self.segment_size:
                    continue

                # append it to the list
                segments.append(new_heart_beat)

            if self.take_average is True:
                segments = np.mean(segments, axis=0)

            all_segments.extend(segments)

        return np.array(all_segments)


class RandomPairedHeartBeatSegmenter(object):
    """Split ecg signal into pairs of heartbeats.

    Args:
        segment_size(tuple or int): Desired segment size.
        take_average(boolean): average the heartbeats pairs
    """

    def __init__(self, segment_size=230, nsegments=8, nrep=100):
        self.segment_size = segment_size
        self.nsegments = nsegments
        self.nrep = nrep

    def __call__(self, sample):
        # find out the R peaks
        r_peak = detect_R_peak(sample)

        sample = sample.cpu().numpy()
        all_segments = []
        half_size_int = int(self.segment_size // 2)

        for recording in range(len(sample)):
            segments = []
            # throw away first and last R peaks as it may lead to segments
            # of len less than desired length.
            for i in r_peak[recording][1: -1]:
                # make a segment
                new_heart_beat = sample[recording][int(
                    i) - int(half_size_int * 0.4): int(i) + int(half_size_int * 1.6)]
                if len(new_heart_beat) != self.segment_size:
                    continue
                # append it to the list
                segments.append(new_heart_beat)

            for i in range(self.nrep):
                random_index = []
                if len(segments) >= self.nsegments:
                    random_index = random.sample(set(range(len(segments))), self.nsegments)
                else:
                    random_index = range(len(segments))

                random_segments = np.array([segments[index] for index in random_index])

                average_segment = np.mean(random_segments, axis=0)

                all_segments.append(average_segment)

        return np.array(all_segments)
