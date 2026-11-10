from .studio import Studio


def configure_project(studio: Studio) -> None:
    """Register team implementations during startup, as they become available."""
    # TODO(llm_engine): Load the shared tokenizer/model and KV cache manager.
    # Set studio.model and studio.device to describe the loaded runtime.
    # Register its unmodified inference as studio.runners["original"].
    # Until then, DSAA_MODEL can enable the built-in local Original adapter.

    # TODO(dsaa_algorithm + llm_engine): Connect the scorer and safe cache
    # eviction to generation, then register studio.runners["dsaa"].
    # Read RunSettings and pass actual KV decisions and scores in InferenceResult.

    # TODO(benchmark baselines): Register studio.runners["h2o"] and
    # studio.runners["snapkv"] when those cache strategies are available.
    # Reuse the same model, tokenizer, and generation settings for comparisons.

    # TODO(evaluation): Assign the evaluation team's BenchmarkEvaluator adapter
    # to studio.evaluator. It provides recall cases, scoring, and aggregation.
