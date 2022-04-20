"""
Inspired from https://huggingface.co/transformers/v3.5.1/_modules/transformers/modeling_utils.html
See there for more info
Most important line for this task is #235
"""

from transformers import top_k_top_p_filtering

import logging

from typing import Iterable, Optional, Tuple
from transformers.generation_utils import GenerationMixin
import torch
from torch import Tensor
from torch.nn import functional as F


def postprocess_next_token_scores(
    model,
    scores,
    input_ids,
    no_repeat_ngram_size,
    bad_words_ids,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    repetition_penalty,
    batch_size,
    num_beams,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        model.enforce_repetition_penalty_(
            scores,
            batch_size,
            num_beams,
            input_ids,
            repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -float("inf")

    # remove endoftext token from generation
    # scores[:, torch.tensor([50256])] = -float("inf")

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = model.calc_banned_ngram_tokens(
            input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len
        )
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -float("inf")

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = model.calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -float("inf")

    return scores


@torch.no_grad()
def custom_generation(
    model,
    device,
    input_ids: Optional[torch.LongTensor] = None,
    max_length: Optional[int] = None,
    min_length: Optional[int] = None,
    do_sample: Optional[bool] = None,
    early_stopping: Optional[bool] = None,
    num_beams: Optional[int] = None,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: Optional[float] = None,
    bad_words_ids: Optional[Iterable[int]] = None,
    bos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    eos_token_id: Optional[int] = None,
    length_penalty: Optional[float] = None,
    no_repeat_ngram_size: Optional[int] = None,
    num_return_sequences: Optional[int] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    decoder_start_token_id: Optional[int] = None,
    use_cache: Optional[bool] = None,
):
    """Generate sequences for each example without beam search (num_beams == 1).
    All returned sequence are generated independantly.
    """

    max_length = max_length if max_length is not None else model.config.max_length
    min_length = min_length if min_length is not None else model.config.min_length
    do_sample = do_sample if do_sample is not None else model.config.do_sample
    early_stopping = (
        early_stopping if early_stopping is not None else model.config.early_stopping
    )
    use_cache = use_cache if use_cache is not None else model.config.use_cache
    num_beams = num_beams if num_beams is not None else model.config.num_beams
    temperature = temperature if temperature is not None else model.config.temperature
    top_k = top_k if top_k is not None else model.config.top_k
    top_p = top_p if top_p is not None else model.config.top_p
    repetition_penalty = (
        repetition_penalty
        if repetition_penalty is not None
        else model.config.repetition_penalty
    )
    bos_token_id = (
        bos_token_id if bos_token_id is not None else model.config.bos_token_id
    )
    pad_token_id = (
        pad_token_id if pad_token_id is not None else model.config.pad_token_id
    )
    eos_token_id = (
        eos_token_id if eos_token_id is not None else model.config.eos_token_id
    )
    length_penalty = (
        length_penalty if length_penalty is not None else model.config.length_penalty
    )
    no_repeat_ngram_size = (
        no_repeat_ngram_size
        if no_repeat_ngram_size is not None
        else model.config.no_repeat_ngram_size
    )
    bad_words_ids = (
        bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
    )
    num_return_sequences = (
        num_return_sequences
        if num_return_sequences is not None
        else model.config.num_return_sequences
    )
    decoder_start_token_id = (
        decoder_start_token_id
        if decoder_start_token_id is not None
        else model.config.decoder_start_token_id
    )

    if input_ids is not None:
        batch_size = input_ids.shape[0]  # overriden by the input batch_size
    else:
        batch_size = 1

    # create attention mask if necessary
    # TODO (PVP): this should later be handled by the forward fn() in each model in the future see PR 3140
    if (
        (attention_mask is None)
        and (pad_token_id is not None)
        and (pad_token_id in input_ids)
    ):
        attention_mask = input_ids.ne(pad_token_id).long()
    elif attention_mask is None:
        attention_mask = input_ids.new_ones(input_ids.shape)

    # length of generated sentences / unfinished sentences
    unfinished_sents = input_ids.new(batch_size).fill_(1)
    sent_lengths = input_ids.new(batch_size).fill_(max_length)

    past = None

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if model.config.is_encoder_decoder:
        if decoder_start_token_id is None:
            decoder_start_token_id = bos_token_id

    if model.config.is_encoder_decoder:
        # create empty decoder_input_ids
        input_ids = torch.full(
            (effective_batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(model.parameters()).device,
        )
        cur_len = 1

        assert (
            batch_size == encoder_outputs[0].shape[0]
        ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} "

        # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and num_return_sequences > 1)
        expanded_batch_idxs = (
            torch.arange(batch_size)
            .view(-1, 1)
            .repeat(1, num_beams * effective_batch_mult)
            .view(-1)
            .to(input_ids.device)
        )
        # expand encoder_outputs
        encoder_outputs = (
            encoder_outputs[0].index_select(0, expanded_batch_idxs),
            *encoder_outputs[1:],
        )

    else:
        encoder_outputs = None
        cur_len = input_ids.shape[-1]

    while cur_len < max_length:
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache
        )

        outputs = model(**model_inputs)
        next_token_logits = outputs[0][:, -1, :]

        scores = postprocess_next_token_scores(
            model,
            scores=next_token_logits,
            input_ids=input_ids,
            no_repeat_ngram_size=no_repeat_ngram_size,
            bad_words_ids=bad_words_ids,
            cur_len=cur_len,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            batch_size=batch_size,
            num_beams=1,
        )

        if do_sample:
            # Temperature (higher temperature => more likely to sample low probability tokens)
            if temperature != 1.0:
                scores = scores / temperature
            # Top-p/top-k filtering
            next_token_logscores = top_k_top_p_filtering(
                scores, top_k=top_k, top_p=top_p
            )
            # Sample
            probs = F.softmax(next_token_logscores, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1)

        # update generations and finished sentences
        if eos_token_id is not None:
            # pad finished sentences if eos_token_id exist
            tokens_to_add = next_token * unfinished_sents + (pad_token_id) * (
                1 - unfinished_sents
            )
        else:
            tokens_to_add = next_token

        # add token and increase length by one
        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        cur_len = cur_len + 1

        if eos_token_id is not None:
            eos_in_sents = tokens_to_add == eos_token_id
            # if sentence is unfinished and the token to add is eos, sent_lengths is filled with current length
            is_sents_unfinished_and_token_to_add_is_eos = unfinished_sents.mul(
                eos_in_sents.long()
            ).bool()
            sent_lengths.masked_fill_(
                is_sents_unfinished_and_token_to_add_is_eos, cur_len
            )
            # unfinished_sents is set to zero if eos in sentence
            unfinished_sents.mul_((~eos_in_sents).long())

        # stop when there is a </s> in each sentence, or if we exceed the maximul length
        if unfinished_sents.max() == 0:
            break

        # [198] == '\n'
        if tokens_to_add == torch.tensor([198]).to(device):
            break

        # extend attention_mask for new generated input if only decoder
        if model.config.is_encoder_decoder is False:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))],
                dim=-1,
            )

    return input_ids
