"""Methods for training DeepSpeech language model."""
import logging
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

import networkx as nx
import rhasspynlu

_LOGGER = logging.getLogger("rhasspyasr_deepspeech")


def train(
    graph: nx.DiGraph,
    language_model: typing.Union[str, Path],
    scorer_path: typing.Union[str, Path],
    alphabet_path: typing.Union[str, Path],
    vocab_path: typing.Optional[typing.Union[str, Path]] = None,
    language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    base_language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    base_language_model_weight: typing.Optional[float] = None,
    mixed_language_model_fst: typing.Optional[typing.Union[str, Path]] = None,
    balance_counts: bool = True,
):
    """Re-generates language model and trie from intent graph"""
    # Language model mixing
    base_fst_weight = None
    if (
        (base_language_model_fst is not None)
        and (base_language_model_weight is not None)
        and (base_language_model_weight > 0)
    ):
        base_fst_weight = (base_language_model_fst, base_language_model_weight)

    with tempfile.TemporaryDirectory() as tmp_folder:
        # Begin training
        with tempfile.NamedTemporaryFile(mode="w+") as arpa_file:
            # 1. Create language model
            _LOGGER.debug("Converting to ARPA language model")
            _LOGGER.debug("With: arpa: %s | modelpath: %s | base_fst_weight: %s | merge_path: %s | vocab_path: %s", arpa_file.name, language_model_fst, base_fst_weight, mixed_language_model_fst, vocab_path)
            if(vocab_path == None):
                vocab_path = str(tmp_folder) + "/vocab-500000.txt"
            
            rhasspynlu.arpa_lm.graph_to_arpa(
                graph,
                arpa_file.name,
                model_path=language_model_fst,
                base_fst_weight=base_fst_weight,
                merge_path=mixed_language_model_fst,
                vocab_path=vocab_path,
            )


        _LOGGER.debug("Converting to vocab to binary language model")
        _LOGGER.debug("With: arpa_file: %s | tmp_folder: %s ", arpa_file.name, tmp_folder)
        vocab_to_binary(vocab_path, tmp_folder)


        with tempfile.NamedTemporaryFile(mode="w+") as scorer_file:
            # 3. Generate trie
            make_scorer(vocab_path, scorer_file.name, alphabet_path, tmp_folder)

            scorer_file.seek(0)
            shutil.copy(scorer_file.name, scorer_path)
            _LOGGER.debug("Wrote scorer to %s", scorer_path)


def vocab_to_binary(
    vocab_path: typing.Union[str, Path], lm_binary_dir: typing.Union[str, Path]
):
    """Convert vocab language model to binary format using kenlm."""
    binary_command = [
        "python", 
        "/opt/rhasspy-asr-deepspeech-hermes/tools/generate_lm.py",
        "--input_txt",
        str(vocab_path),
        "--output_dir", 
        str(lm_binary_dir),
        "--top_k", 
        "500000", 
        "--kenlm_bins", 
        "/opt/rhasspy-asr-deepspeech-hermes/tools/kenlm", 
        "--arpa_order", 
        "5", 
        "--max_arpa_memory", 
        "50%",
        "--arpa_prune", 
        "0|0|1", 
        "--binary_a_bits", 
        "255", 
        "--binary_q_bits", 
        "8", 
        "--binary_type", 
        "trie", 
        "--discount_fallback"
    ]
    _LOGGER.debug(binary_command)
    subprocess.check_call(binary_command)

def make_scorer(
    vocab_path: typing.Union[str, Path],
    scorer_file: typing.Union[str, Path],
    alphabet_path: typing.Union[str, Path],
    tmp_folder: typing.Union[str, Path]
):
    """Generate trie using Mozilla native-client tool."""

    _LOGGER.debug("Create Scorerfile")
    _LOGGER.debug("With: vocab_path: %s | scorer_file: %s | alphabet_path: %s | lm_binary_dir: %s", vocab_path, scorer_file, alphabet_path, tmp_folder)
     
    trie_command = [
        "python3", 
        "/opt/rhasspy-asr-deepspeech-hermes/tools/generate_package.py", 
        "--alphabet", 
        str(alphabet_path), 
        "--lm", 
        str(tmp_folder) + "/lm.binary", 
        "--vocab",
        str(vocab_path), 
        "--package", 
        str(scorer_file), 
        "--default_alpha", 
        "0.931289039105002", 
        "--default_beta", 
        "1.1834137581510284"
    ]

    _LOGGER.debug(trie_command)
    subprocess.check_call(trie_command)
