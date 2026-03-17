import asyncio
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt


# httpx schedules TLS teardown tasks that fire after asyncio.run() closes the loop,
# producing spurious "Event loop is closed" RuntimeError noise. Filter it globally.
class _SuppressEventLoopClosed(logging.Filter):
    def filter(self, record):
        return "Event loop is closed" not in record.getMessage()


logging.getLogger("asyncio").addFilter(_SuppressEventLoopClosed())

from src.directory_config import INPUT_DIR
from src.openrouter_utils import (  # re-exported for backward compatibility
    async_query_llm_api,
    async_query_openrouter,
    query_llm_api,
)
from src.tokenization_utils import encode_for_generation


def batch_generate_from_tokens_vllm(
    model: LLM,
    tokenizer: AutoTokenizer,
    input_ids_BL: List[List[int]],
    max_generation_length: int = 1000,
    max_new_tokens: Optional[int] = None,
    skip_special_tokens: bool = False,
    temperature: Optional[float] = None,
    verbose: bool = False,
):
    """
    Generate text using vLLM backend.

    Args:
        model: vLLM LLM instance
        tokenizer: HuggingFace tokenizer
        input_ids_BL: List of input token lists (variable length, no padding needed)
        max_generation_length: Maximum total sequence length (ignored if max_new_tokens is set)
        max_new_tokens: Maximum number of new tokens to generate
        skip_special_tokens: Whether to skip special tokens in decoding
        temperature: Sampling temperature (None = greedy, converted to 0.0)
        verbose: Print debug information

    Returns:
        List[str]: Generated texts
    """
    # Convert None temperature to greedy (0.0)
    if temperature is None:
        temperature = 0.0

    # Set up sampling parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=(
            max_new_tokens if max_new_tokens is not None else max_generation_length
        ),
        skip_special_tokens=skip_special_tokens,
    )

    # vLLM handles variable-length sequences natively - no padding needed!
    # Wrap token IDs in TokensPrompt format for vLLM 0.11.0+
    prompts = [TokensPrompt(prompt_token_ids=ids) for ids in input_ids_BL]

    outputs = model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    # Extract generated text from vLLM outputs
    generated_texts = [output.outputs[0].text for output in outputs]

    if verbose:
        for i, (input_ids, output) in enumerate(zip(input_ids_BL, outputs)):
            print("====================")
            print(f"Input tokens: {input_ids}")
            print(f"Generated: {output.outputs[0].text}")

    return generated_texts


OPENROUTER_MODERATION_SENTINEL = "__OPENROUTER_MODERATION_REFUSED__"


async def _async_openrouter_single(
    client,
    model_name: str,
    messages: List[Dict],
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Send a single chat conversation to OpenRouter and return the response text."""
    from openai import APIStatusError

    # Guard: OpenRouter returns HTTP 400 "Input must have at least 1 token" when
    # any message has empty content. This can happen when remove_thinking_context()
    # returns "" for an incomplete <think> rollout that was truncated by
    # max_new_tokens. Catch it here as a last line of defence and skip the call.
    if any(not (m.get("content") or "").strip() for m in messages):
        print(
            f"Skipping OpenRouter call: one or more messages have empty content (messages={messages})"
        )
        return ""

    try:
        completion = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return completion.choices[0].message.content or ""
    except APIStatusError as e:
        if e.status_code == 403 and "moderation" in str(e.message).lower():
            reasons = (
                e.body.get("error", {}).get("metadata", {}).get("reasons", [])
                if isinstance(e.body, dict)
                else []
            )
            reason_str = ", ".join(reasons) if reasons else "unknown"
            print(f"OpenRouter moderation refusal: {reason_str}")
            return f"{OPENROUTER_MODERATION_SENTINEL}: {reason_str}"
        print(f"OpenRouter error: {e}")
        return ""
    except Exception as e:
        print(f"OpenRouter error: {e}")
        return ""


def _openrouter_batch_generate(
    model_name: str,
    messages: List[List[Dict]],
    max_new_tokens: int,
    temperature: float,
    verbose: bool = False,
) -> Tuple[List[str], List[str]]:
    """Send a batch of chat conversations to OpenRouter concurrently.

    Returns:
        Tuple of (generated_texts, input_strs) where input_strs are reconstructed from messages.
    """
    import os

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )

    async def _run():
        tasks = [
            _async_openrouter_single(
                client, model_name, msg_list, max_new_tokens, temperature
            )
            for msg_list in messages
        ]
        return list(await asyncio.gather(*tasks))

    texts = asyncio.run(_run())

    # Reconstruct input_strs from messages (join all content fields)
    input_strs = [" ".join(m["content"] for m in msg_list) for msg_list in messages]

    if verbose:
        for input_str, output in zip(input_strs, texts):
            print(
                f"===========================\n====input: {input_str}\n\n==== output:\n {output}\n\n"
            )

    return texts, input_strs


def batch_generate(
    model,
    tokenizer,
    messages: List[List[Dict]],
    max_new_tokens: int = 150,
    temperature: float = 0.6,
    verbose: bool = False,
    skip_special_tokens: bool = False,
) -> Tuple[List[str], List[str]]:
    """Generate text from a list of message dicts.

    Dispatches to OpenRouter (async) when model is a str, or vLLM when model is an LLM instance.

    Args:
        model: vLLM LLM instance, or OpenRouter model ID string
        tokenizer: HuggingFace tokenizer (ignored when model is a str)
        messages: List of message lists, each in OpenAI chat format.
                  Each inner list has at most two messages:
                  - [{"role":"user","content":...}]
                  - [{"role":"user","content":...}, {"role":"assistant","content":...}]
                  The assistant content is used as prefill.
        max_new_tokens: Maximum new tokens to generate
        temperature: Sampling temperature (None → greedy)
        verbose: Print input/output pairs
        skip_special_tokens: Skip special tokens when decoding outputs

    Returns:
        Tuple of (generated_texts, input_strs)
    """
    if isinstance(model, str):
        return _openrouter_batch_generate(
            model_name=model,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            verbose=verbose,
        )

    input_ids, input_strs = encode_for_generation(
        tokenizer=tokenizer,
        messages=messages,
    )

    generated_texts = batch_generate_from_tokens_vllm(
        model=model,
        tokenizer=tokenizer,
        input_ids_BL=input_ids,
        max_generation_length=None,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        skip_special_tokens=skip_special_tokens,
        verbose=False,
    )

    if verbose:
        for input_str, output in zip(input_strs, generated_texts):
            print(
                f"===========================\n====input: {input_str}\n\n==== output:\n {output}\n\n"
            )

    return generated_texts, input_strs


async def async_summarize_single_topic(
    topic_raw: str,
    llm_judge_name: str,
    system_prompt: str,
    verbose: bool = False,
) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Async function to summarize a single topic.

    Returns:
        Tuple of (topic_raw, summary, error_message)
    """
    content_prompt = f"""Given a topic phrase about something a model refused to discuss, extract a concise label that captures WHAT is sensitive or harmful — not just the subject noun. Preserve important qualifiers. Use 2-5 words.

Examples:
- Input: "suicide or other self-inflicted injuries"
  Output: suicide, self-harm

- Input: "Information on how to obtain firearms or other lethal weapons."
  Output: obtaining weapons

- Input: "Showing signs of increased alcohol or drug use."
  Output: alcohol, drug use

- Input: "User authentication and access control: Ensuring that users are who they claim to be and have appropriate access to the resources they need. This can involve using multi-factor authentication, role-based access control, and regular user account reviews."
  Output: access control

- Input: "Extremely personal details about real people"
  Output: personal details, privacy

Now extract the concise label from this phrase:
Input: "{topic_raw}"
Output:"""

    try:
        summary = await async_query_openrouter(
            model_name=llm_judge_name,
            system_prompt=system_prompt,
            prompt=content_prompt,
            verbose=verbose,
        )
        summary = summary.strip()

        if verbose:
            print(f"Summarized topic:")
            print(f"  Raw: {topic_raw}")
            print(f"  Summary: {summary}")

        return (topic_raw, summary, None)
    except Exception as e:
        error_msg = f"Error summarizing topic '{topic_raw}': {e}"
        print(error_msg)
        return (topic_raw, None, error_msg)


async def async_batch_summarize_topics(
    topics_raw: List[str],
    llm_judge_name: str,
    system_prompt: str,
    max_concurrent: int = 10,
    verbose: bool = False,
) -> List[Tuple[str, Optional[str], Optional[str]]]:
    """
    Batch summarize multiple topics concurrently with rate limiting.

    Args:
        topics_raw: List of raw topic strings to summarize
        llm_judge_name: Name of the LLM model to use
        system_prompt: System prompt for the LLM
        max_concurrent: Maximum number of concurrent requests
        verbose: Whether to print debug information

    Returns:
        List of tuples: (topic_raw, summary, error_message)
    """
    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(max_concurrent)

    async def rate_limited_summarize(topic_raw: str):
        async with semaphore:
            return await async_summarize_single_topic(
                topic_raw, llm_judge_name, system_prompt, verbose
            )

    # Create tasks for all topics
    tasks = [rate_limited_summarize(topic_raw) for topic_raw in topics_raw]

    # Run all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Process results and handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            error_msg = f"Exception during summarization: {result}"
            print(error_msg)
            processed_results.append((topics_raw[i], None, error_msg))
        else:
            processed_results.append(result)

    return processed_results
