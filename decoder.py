import torch
from six.moves import xrange
from torchaudio.models.decoder import ctc_decoder


class Decoder(object):
    """
    Basic decoder class from which all other decoders inherit. Implements several
    helper functions. Subclasses should implement the decode() method.

    Arguments:
        labels (list): mapping from integers to characters.
        blank_index (int, optional): index for the blank '_' character. Defaults to 0.
    """

    def __init__(self, labels, blank_index=0):
        self.labels = labels
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in labels:
            space_index = labels.index(' ')
        self.space_index = space_index

    def decode(self, probs, sizes=None):
        """
        Given a matrix of character probabilities, returns the decoder's
        best guess of the transcription

        Arguments:
            probs: Tensor of character probabilities, where probs[c,t]
                            is the probability of character c at time t
            sizes(optional): Size of each sequence in the mini-batch
        Returns:
            string: sequence of the model's best guess for the transcription
        """
        raise NotImplementedError


class BeamCTCDecoder(Decoder):
    def __init__(
        self,
        labels,
        beam_size=100,
        blank_index=0,
        nbest=1
    ):
        super(BeamCTCDecoder, self).__init__(labels)
        labels = list(labels)  # Ensure labels are a list before passing to decoder
        self._decoder = ctc_decoder(
            lexicon=None,
            tokens=labels,
            nbest=nbest,
            beam_size=beam_size,
            beam_threshold=100,
            blank_token=labels[blank_index],
            sil_token=labels[blank_index],
            unk_word=labels[blank_index]
        )

    def decode(self, probs, sizes=None, return_all=False):
        probs = probs.detach().cpu()
        sizes = sizes.detach().cpu()
        offsets = []
        strings = []
        for sample in self._decoder(probs, sizes):
            if return_all:
                ctc_hypophesis = sample
                o = []
                s = []
                for h in ctc_hypophesis:
                    idxes = h.tokens
                    o.append(idxes)
                    s.append(''.join(self._decoder.idxs_to_tokens(idxes)))
                offsets.append(o)
                strings.append(s)
            else:
                ctc_hypophesis = sample[0]
                idxes = ctc_hypophesis.tokens
                offsets.append(idxes)
                strings.append(''.join(self._decoder.idxs_to_tokens(idxes)))
        return strings, offsets

    def convert_to_strings(self, sequences):
        strings = []
        for idxes in sequences:
            strings.append(''.join(self._decoder.idxs_to_tokens(idxes)))
        return strings


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super(GreedyDecoder, self).__init__(labels, blank_index)

    def convert_to_strings(self,
                           sequences,
                           sizes=None,
                           remove_repetitions=False,
                           return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in xrange(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self,
                       sequence,
                       size,
                       remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)),
                                                   sizes,
                                                   remove_repetitions=True,
                                                   return_offsets=True)
        strings = [s[0] for s in strings]
        return strings, offsets
