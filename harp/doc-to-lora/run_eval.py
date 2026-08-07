import logging

from ctx_to_lora.eval_utils import run_eval

logger = logging.getLogger()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Evaluate a base model from HuggingFace Hub, without loading checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help=(
            "Specific datasets to evaluate on."
            "If not provided, uses default from args.yaml"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Eval batch size for teacher forcing",
    )
    parser.add_argument(
        "--eval_batch_size_gen",
        type=int,
        default=32,
        help="Eval batch size for generation",
    )
    parser.add_argument(
        "--max_val_samples_per_ds",
        type=int,
        default=-1,
        help=(
            "Maximum number of validation samples per dataset. "
            "If -1, uses values from checkpoint config."
        ),
    )
    parser.add_argument(
        "--max_test_samples_per_ds",
        type=int,
        default=500,
        help=(
            "Maximum number of validation samples per dataset. "
            "If -1, uses values from checkpoint config."
        ),
    )
    parser.add_argument(
        "--max_ctx_chunk_len",
        type=int,
        default=-1,
        help="Maximum length of context chunk for evaluation",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate during evaluation",
    )
    parser.add_argument(
        "--remove_context",
        action="store_true",
        help="Remove context when evaluating the base model.",
    )
    parser.add_argument(
        "--use_cd",
        action="store_true",
        help="Use context distillation model for evaluation.",
    )
    parser.add_argument(
        "--cd_update_iterations",
        type=int,
        default=20,
        help="Number of update iterations for context distillation during evaluation",
    )
    parser.add_argument(
        "--cd_use_gen_q",
        action="store_true",
        help="Use generated queries for context distillation training.",
    )
    parser.add_argument(
        "--q_gen_rounds",
        type=int,
        default=4,
        help="Number of rounds of query generation for context distillation.",
    )
    parser.add_argument(
        "--cd_batch_size",
        type=int,
        default=16,
        help="Batch size for context distillation.",
    )
    parser.add_argument(
        "--use_iterative_mode",
        action="store_true",
        help="Use iterative mode LoRA layer-by-layer generation",
    )
    parser.add_argument(
        "--use_llmlingua",
        action="store_true",
        help="Use LLMLingua compression for evaluation",
    )
    parser.add_argument(
        "--llmlingua_compression_rate",
        type=float,
        default=0.9,
        help="Compression rate for LLMLingua",
    )
    parser.add_argument(
        "--use_t2l",
        action="store_true",
        help="Use Text-to-LoRA model for evaluation",
    )
    parser.add_argument(
        "--add_ctx_to_input",
        action="store_true",
        help="Add ctx to base model's input",
    )
    parser.add_argument(
        "--truncate_if_too_long_inp",
        action="store_true",
        help="Truncate input sequences that are too long",
    )
    parser.add_argument(
        "--truncate_if_too_long_ctx",
        action="store_true",
        help="Truncate ctx sequences that are too long",
    )
    parser.add_argument(
        "--gen_lora_scaling",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--flip_ctx_inp",
        action="store_true",
        help="Flip the order of context and input",
    )
    parser.add_argument(
        "--use_generative_adapter",
        action="store_true",
        help="Use generative adapter for evaluation",
    )

    cli_args = vars(parser.parse_args())

    if cli_args["model_name_or_path"]:
        assert cli_args["max_ctx_chunk_len"] <= 0, (
            f"Evaluating base model shouldn't be used with `max_ctx_chunk_len`"
        )

    eval_batch_size_gen = cli_args.pop("eval_batch_size_gen")
    eval_batch_size = cli_args.pop("eval_batch_size")
    run_eval(
        **cli_args,
        eval_batch_size=eval_batch_size_gen,
        generative=True,
    )
