"""Representations Extractor for ``transformers`` toolkit models.

Module that given a file with input sentences and a ``transformers``
model, extracts representations from all layers of the model. The script
supports aggregation over sub-words created due to the tokenization of
the provided model.

Can also be invoked as a script as follows:
    ``python -m neurox.data.extraction.transformers_extractor``
"""

import argparse
import sys

import numpy as np
import torch
import pandas as pd

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification

from neurox.data.writer import ActivationsWriter

def get_model_and_tokenizer(model_desc, device="cpu", random_weights=False):
    """
    Automatically get the appropriate ``transformers`` model and tokenizer based
    on the model description

    Parameters
    ----------
    model_desc : str
        Model description; can either be a model name like ``bert-base-uncased``,
        a comma separated list indicating <model>,<tokenizer> (since 1.0.8),
        or a path to a trained model

    device : str, optional
        Device to load the model on, cpu or gpu. Default is cpu.

    random_weights : bool, optional
        Whether the weights of the model should be randomized. Useful for analyses
        where one needs an untrained model.

    Returns
    -------
    model : transformers model
        An instance of one of the transformers.modeling classes
    tokenizer : transformers tokenizer
        An instance of one of the transformers.tokenization classes
    """
    model_desc = model_desc.split(",")
    if len(model_desc) == 1:
        model_name = model_desc[0]
        tokenizer_name = model_desc[0]
    else:
        model_name = model_desc[0]
        tokenizer_name = model_desc[1]
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if random_weights:
        print("Randomizing weights")
        model.init_weights()

    return model, tokenizer


def aggregate_repr(state, start, end, aggregation):
    """
    Function that aggregates activations/embeddings over a span of subword tokens.
    This function will usually be called once per word. For example, if we had the sentence::

        This is an example

    which is tokenized by BPE into::

        this is an ex @@am @@ple

    The function should be called 4 times::

        aggregate_repr(state, 0, 0, aggregation)
        aggregate_repr(state, 1, 1, aggregation)
        aggregate_repr(state, 2, 2, aggregation)
        aggregate_repr(state, 3, 5, aggregation)

    Returns a zero vector if end is less than start, i.e. the request is to
    aggregate over an empty slice.

    Parameters
    ----------
    state : numpy.ndarray
        Matrix of size [ NUM_LAYERS x NUM_SUBWORD_TOKENS_IN_SENT x LAYER_DIM]
    start : int
        Index of the first subword of the word being processed
    end : int
        Index of the last subword of the word being processed
    aggregation : {'first', 'last', 'average'}
        Aggregation method for combining subword activations

    Returns
    -------
    word_vector : numpy.ndarray
        Matrix of size [NUM_LAYERS x LAYER_DIM]
    """
    if end < start:
        sys.stderr.write("WARNING: An empty slice of tokens was encountered. " +
            "This probably implies a special unicode character or text " +
            "encoding issue in your original data that was dropped by the " +
            "transformer model's tokenizer.\n")
        return np.zeros((state.shape[0], state.shape[2]))
    if aggregation == "first":
        return state[:, start, :]
    elif aggregation == "last":
        return state[:, end, :]
    elif aggregation == "average":
        return np.average(state[:, start : end + 1, :], axis=1)


def extract_sentence_representations(
    sentence,
    model,
    tokenizer,
    device="cpu",
    include_embeddings=True,
    aggregation="last",
    tokenization_counts={}
):
    """
    Get representations for one sentence
    """
    # this follows the HuggingFace API for transformers

    special_tokens = [
        x for x in tokenizer.all_special_tokens if x != tokenizer.unk_token
    ]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)

    original_tokens = sentence.split(" ")
    # Add a letter and space before each word since some tokenizers are space sensitive
    tmp_tokens = [
        "a" + " " + x if x_idx != 0 else x for x_idx, x in enumerate(original_tokens)
    ]
    assert len(original_tokens) == len(tmp_tokens)

    with torch.no_grad():
        # Get tokenization counts if not already available
        for token_idx, token in enumerate(tmp_tokens):
            tok_ids = [
                x for x in tokenizer.encode(token) if x not in special_tokens_ids
            ]
            if token_idx != 0:
                # Ignore the first token (added letter)
                tok_ids = tok_ids[1:]

            if token in tokenization_counts:
                assert tokenization_counts[token] == len(
                    tok_ids
                ), "Got different tokenization for already processed word"
            else:
                tokenization_counts[token] = len(tok_ids)
        ids = tokenizer.encode(sentence, truncation=True)
        input_ids = torch.tensor([ids]).to(device)
        # Hugging Face format: tuple of torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)
        # Tuple has 13 elements for base model: embedding outputs + hidden states at each layer
        all_hidden_states = model(input_ids)[-1]

        if include_embeddings:
            all_hidden_states = [
                hidden_states[0].cpu().numpy() for hidden_states in all_hidden_states
            ]
        else:
            all_hidden_states = [
                hidden_states[0].cpu().numpy()
                for hidden_states in all_hidden_states[1:]
            ]
        all_hidden_states = np.array(all_hidden_states)

    print('Sentence         : "%s"' % (sentence))
    print("Original    (%03d): %s" % (len(original_tokens), original_tokens))
    print(
        "Tokenized   (%03d): %s"
        % (
            len(tokenizer.convert_ids_to_tokens(ids)),
            tokenizer.convert_ids_to_tokens(ids),
        )
    )

    # Remove special tokens
    ids_without_special_tokens = [x for x in ids if x not in special_tokens_ids]
    idx_without_special_tokens = [
        t_i for t_i, x in enumerate(ids) if x not in special_tokens_ids
    ]
    filtered_ids = [ids[t_i] for t_i in idx_without_special_tokens]
    assert all_hidden_states.shape[1] == len(ids)
    all_hidden_states = all_hidden_states[:, idx_without_special_tokens, :]
    assert all_hidden_states.shape[1] == len(filtered_ids)
    print(
        "Filtered   (%03d): %s"
        % (
            len(tokenizer.convert_ids_to_tokens(filtered_ids)),
            tokenizer.convert_ids_to_tokens(filtered_ids),
        )
    )
    segmented_tokens = tokenizer.convert_ids_to_tokens(filtered_ids)

    # Perform actual subword aggregation/detokenization
    counter = 0
    detokenized = []
    final_hidden_states = np.zeros(
        (all_hidden_states.shape[0], len(original_tokens), all_hidden_states.shape[2])
    )
    inputs_truncated = False

    for token_idx, token in enumerate(tmp_tokens):
        current_word_start_idx = counter
        current_word_end_idx = counter + tokenization_counts[token]

        # Check for truncated hidden states in the case where the
        # original word was actually tokenized
        if  (tokenization_counts[token] != 0 and current_word_start_idx >= all_hidden_states.shape[1]) \
                or current_word_end_idx > all_hidden_states.shape[1]:
            final_hidden_states = final_hidden_states[:, :len(detokenized), :]
            inputs_truncated = True
            break

        final_hidden_states[:, len(detokenized), :] = aggregate_repr(
            all_hidden_states,
            current_word_start_idx,
            current_word_end_idx - 1,
            aggregation,
        )
        detokenized.append(
            "".join(segmented_tokens[current_word_start_idx:current_word_end_idx])
        )
        counter += tokenization_counts[token]

    print("Detokenized (%03d): %s" % (len(detokenized), detokenized))
    print("Counter: %d" % (counter))

    if inputs_truncated:
        print("WARNING: Input truncated because of length, skipping check")
    else:
        assert counter == len(ids_without_special_tokens)
        assert len(detokenized) == len(original_tokens)
    print("===================================================================")

    return final_hidden_states, detokenized


def extract_representations(
    model,
    input_tokens_list,  # List of tokenized input lists
    output_file,
    device="cpu",
    output_type="json",
    decompose_layers=False,
    filter_layers=None,
):
    print("📤 Saving activations to file")
    writer = ActivationsWriter.get_writer(
        output_file,
        filetype=output_type,
        decompose_layers=decompose_layers,
        filter_layers=filter_layers,
    )

    print("🔄 Extracting CLS token representations from all hidden layers")
    for sentence_idx, token_list in enumerate(input_tokens_list):
        input_ids = torch.tensor([token_list]).to(device)  # (1, seq_len)

        with torch.no_grad():
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states[1:]  # Skip embedding layer

        hidden_states = torch.stack(hidden_states).squeeze(1)  # (num_layers, seq_len, hidden_dim)

        cls_hidden_states = hidden_states[:, 0, :]            # (num_layers, hidden_dim)
        cls_hidden_states = cls_hidden_states.unsqueeze(1)   # (num_layers, 1, hidden_dim)

        writer.write_activations(
            sentence_idx,
            ["[CLS]"],
            cls_hidden_states.cpu().numpy()
        )

        print(f"✅ Sample {sentence_idx}: CLS activations extracted with shape {cls_hidden_states.shape}")

    writer.close()
    print(f"✅ Activations saved to {output_file}")





HDF5_SPECIAL_TOKENS = {".": "__DOT__", "/": "__SLASH__"}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_desc", help="Model name or path")
    parser.add_argument("input_csv", help="CSV file with input_ids")
    parser.add_argument("output_file", help="Output file for activations")
    parser.add_argument("--disable_cuda", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--random_weights", action="store_true", help="Skip loading weights")

    ActivationsWriter.add_writer_options(parser)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.disable_cuda else "cpu")

    print(f"📥 Loading model from {args.model_desc}")
    model = AutoModelForSequenceClassification.from_pretrained(args.model_desc, num_labels=5)
    
    weights_path = "/home/kikay/LLM_Syscalls/best_model_BigBird.pth"
    print(f"🔄 Loading weights from {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()
    print("✅ Model loaded and ready")

    print(f"📥 Loading data from {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    df['input_ids'] = df['input_ids'].apply(lambda x: list(map(int, x.strip("[]").split(","))))
    print("✅ Data loaded successfully")

    extract_representations(
        model,
        df["input_ids"].tolist(),
        args.output_file,
        device=device,
        output_type=args.output_filetype,
        decompose_layers=args.decompose_layers,
        filter_layers=args.filter_layers
    )

    print(f"✅ Activations extracted and saved to {args.output_file}")
