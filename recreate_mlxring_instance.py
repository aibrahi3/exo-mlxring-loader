#!/usr/bin/env python3
import argparse
import difflib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def fetch_json(url: str, timeout: int = 30):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def request_json(url: str, method: str = "GET", payload=None, timeout: int = 30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    if not body:
        return None
    return json.loads(body.decode())


def get_instance_parts(instance_wrapper):
    instance_kind, inner = next(iter(instance_wrapper.items()))
    shard_assignments = inner.get("shardAssignments", {})
    model_id = shard_assignments.get("modelId")
    instance_id = inner.get("instanceId")
    return model_id, instance_kind, instance_id, inner


def list_models(base_url: str):
    response = request_json(f"{base_url}/v1/models")
    data = response.get("data", []) if response else []
    return sorted(
        item["id"]
        for item in data
        if isinstance(item, dict) and isinstance(item.get("id"), str) and item["id"]
    )


def list_instance_models(state):
    models = set()
    for wrapper in state.get("instances", {}).values():
        model_id, _, _, _ = get_instance_parts(wrapper)
        if model_id:
            models.add(model_id)
    return sorted(models)


def find_instances_for_model(state, model_id: str):
    matches = []
    for instance_id, wrapper in state.get("instances", {}).items():
        found_model_id, instance_kind, _, inner = get_instance_parts(wrapper)
        if found_model_id == model_id:
            matches.append(
                {
                    "instance_id": instance_id,
                    "instance_kind": instance_kind,
                    "inner": inner,
                }
            )
    return matches


def get_instance_by_id(state, instance_id: str):
    wrapper = (state.get("instances") or {}).get(instance_id)
    if not wrapper:
        return None
    return wrapper


def choose_preview(previews, preferred_meta="MlxRing", preferred_sharding="Pipeline"):
    preferred = []
    fallback = []
    for preview in previews:
        if preview.get("instance") is None:
            continue
        if (
            preview.get("instance_meta") == preferred_meta
            and preview.get("sharding") == preferred_sharding
        ):
            preferred.append(preview)
        elif preview.get("instance_meta") == preferred_meta:
            fallback.append(preview)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def runner_ids_for_instance(instance_wrapper):
    _, _, _, inner = get_instance_parts(instance_wrapper)
    return list(inner.get("shardAssignments", {}).get("nodeToRunner", {}).values())


def runner_state_name(runners, runner_id):
    runner = runners.get(runner_id, {})
    if not runner:
        return "Missing"
    return next(iter(runner.keys()))


def all_runners_ready(state, instance_wrapper):
    runner_ids = runner_ids_for_instance(instance_wrapper)
    runners = state.get("runners", {})
    readyish = {"RunnerReady", "RunnerIdle"}
    return bool(runner_ids) and all(
        runner_state_name(runners, runner_id) in readyish for runner_id in runner_ids
    )


def any_runner_failed(state, instance_wrapper):
    runner_ids = runner_ids_for_instance(instance_wrapper)
    runners = state.get("runners", {})
    failures = {}
    for runner_id in runner_ids:
        runner = runners.get(runner_id, {})
        state_name = next(iter(runner.keys())) if runner else "Missing"
        if state_name == "RunnerFailed":
            failures[runner_id] = runner[state_name]
    return failures


def test_completion(base_url: str, model_id: str, prompt: str, timeout: int):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 24,
    }
    return request_json(
        f"{base_url}/v1/chat/completions",
        method="POST",
        payload=payload,
        timeout=timeout,
    )


def summarize_models(models, limit=6):
    if not models:
        return "none"
    if len(models) <= limit:
        return ", ".join(models)
    shown = ", ".join(models[:limit])
    return f"{shown}, ... ({len(models)} total)"


def format_ambiguous_model_error(instance_models, api_models):
    details = []
    if instance_models:
        details.append(f"active instance models: {summarize_models(instance_models)}")
    if api_models:
        details.append(f"/v1/models entries: {summarize_models(api_models)}")
    if not details:
        details.append("no models were discoverable from /state or /v1/models")
    return (
        "Could not auto-resolve a model. "
        "Pass --model <model-id> explicitly or rerun in an interactive terminal to choose one. "
        + " | ".join(details)
    )


def normalize_text(value: str):
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def compact_text(value: str):
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def find_model_matches(query: str, models):
    normalized_query = normalize_text(query)
    compact_query = compact_text(query)
    query_tokens = [token for token in normalized_query.split() if token]

    if not compact_query:
        return []

    scored = []
    for model in models:
        normalized_model = normalize_text(model)
        compact_model = compact_text(model)
        model_tokens = set(normalized_model.split())

        score = 0
        if query.lower() == model.lower():
            score = max(score, 1000)
        if normalized_query and normalized_query == normalized_model:
            score = max(score, 950)
        if compact_query == compact_model:
            score = max(score, 900)
        if compact_query in compact_model:
            score = max(score, 800 + min(len(compact_query), 99))

        if query_tokens:
            overlap = sum(1 for token in query_tokens if token in model_tokens)
            if overlap:
                score = max(score, 500 + overlap * 20)
            if all(token in model_tokens for token in query_tokens):
                score = max(score, 700 + len(query_tokens) * 10)

        ratio = difflib.SequenceMatcher(None, compact_query, compact_model).ratio()
        if ratio >= 0.6:
            score = max(score, int(ratio * 100))

        if score > 0:
            scored.append((score, model))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [model for _, model in scored]


def prompt_for_model(instance_models, api_models):
    print("Multiple models are available, so model selection is needed.")
    if instance_models:
        print("Active instance models:")
        for index, model in enumerate(instance_models, start=1):
            print(f"  {index}. {model}")
    print("Enter a number, paste a model ID, or type search text.")

    last_matches = []
    while True:
        try:
            response = input("Model selection: ").strip()
        except EOFError as exc:
            raise ValueError(
                "Interactive model selection requires a TTY. Pass --model <model-id> explicitly."
            ) from exc
        except KeyboardInterrupt as exc:
            raise ValueError("Model selection cancelled by user.") from exc

        if not response:
            print("Please enter a model number, model ID, or search text.")
            continue

        if response.isdigit():
            selection = int(response)
            if last_matches and 1 <= selection <= len(last_matches):
                return last_matches[selection - 1]
            if instance_models and 1 <= selection <= len(instance_models):
                return instance_models[selection - 1]
            print("That number is not in the current list.")
            continue

        if response in api_models or response in instance_models:
            return response

        exact_lower = {model.lower(): model for model in api_models}
        if response.lower() in exact_lower:
            return exact_lower[response.lower()]

        matches = find_model_matches(response, api_models)
        if len(matches) == 1:
            print(f"Selected: {matches[0]}")
            return matches[0]
        if not matches:
            suggestions = difflib.get_close_matches(
                response, api_models, n=5, cutoff=0.35
            )
            if suggestions:
                print("No exact matches found. Closest matches:")
                for index, model in enumerate(suggestions, start=1):
                    print(f"  {index}. {model}")
                last_matches = suggestions
                print("Enter a number from the list above, or try a different search.")
            else:
                print("No matching models found. Refine the search or paste a model ID.")
            continue

        last_matches = matches[:20]
        print(f"Found {len(matches)} matching models:")
        for index, model in enumerate(last_matches, start=1):
            print(f"  {index}. {model}")
        if len(matches) > len(last_matches):
            print("Showing the first 20 matches.")


def resolve_model_id(explicit_model, state, base_url):
    if explicit_model and explicit_model.lower() != "auto":
        return explicit_model, "explicit --model"

    instance_models = list_instance_models(state)
    if len(instance_models) == 1:
        return instance_models[0], "single active model found in /state"

    api_models = list_models(base_url)
    if len(api_models) == 1:
        return api_models[0], "single model found in /v1/models"

    if sys.stdin.isatty():
        chosen = prompt_for_model(instance_models, api_models)
        return chosen, "interactive selection"

    raise ValueError(format_ambiguous_model_error(instance_models, api_models))


def wait_until_gone(base_url: str, instance_ids, deadline: float, poll_interval: float):
    pending = set(instance_ids)
    while pending and time.time() < deadline:
        state = fetch_json(f"{base_url}/state")
        pending = {iid for iid in pending if iid in (state.get("instances") or {})}
        if pending:
            time.sleep(poll_interval)
    return not pending


def main():
    parser = argparse.ArgumentParser(
        description="Recreate a local EXO model instance as MlxRing and optionally test the API."
    )
    parser.add_argument("--base-url", default="http://localhost:52415")
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID to reload. Omit or pass 'auto' to choose interactively when needed.",
    )
    parser.add_argument("--preferred-sharding", default="Pipeline")
    parser.add_argument("--keep-existing", action="store_true")
    parser.add_argument("--delete-all-instances", action="store_true")
    parser.add_argument("--wait-seconds", type=int, default=180)
    parser.add_argument("--poll-interval", type=float, default=3.0)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--test-prompt", default="Reply with exactly OK.")
    parser.add_argument("--test-timeout", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    state_url = f"{base_url}/state"

    try:
        state = fetch_json(state_url)
    except urllib.error.URLError as exc:
        print(f"ERROR: EXO state endpoint is unreachable at {state_url}: {exc}", file=sys.stderr)
        return 2

    try:
        model_id, model_reason = resolve_model_id(args.model, state, base_url)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3

    existing_model_instances = find_instances_for_model(state, model_id)
    previews_url = (
        f"{base_url}/instance/previews?model_id={urllib.parse.quote(model_id, safe='')}"
    )
    previews = fetch_json(previews_url).get("previews", [])
    chosen = choose_preview(previews, preferred_sharding=args.preferred_sharding)

    print(f"Base URL: {base_url}")
    print(f"Model: {model_id}")
    print(f"Model selection: {model_reason}")
    print(f"Existing instances for model: {len(existing_model_instances)}")
    for item in existing_model_instances:
        print(f"  - {item['instance_id']} ({item['instance_kind']})")

    if not chosen:
        print("ERROR: No usable MlxRing preview was available.", file=sys.stderr)
        for preview in previews:
            print(
                f"  preview meta={preview.get('instance_meta')} sharding={preview.get('sharding')} error={preview.get('error')}",
                file=sys.stderr,
            )
        return 3

    chosen_wrapper = chosen["instance"]
    _, chosen_kind, chosen_instance_id, _ = get_instance_parts(chosen_wrapper)
    print(
        f"Chosen preview: meta={chosen.get('instance_meta')} sharding={chosen.get('sharding')} instance_id={chosen_instance_id}"
    )

    if args.dry_run:
        return 0

    if not args.keep_existing:
        if args.delete_all_instances:
            delete_targets = list((state.get("instances") or {}).keys())
        else:
            delete_targets = [item["instance_id"] for item in existing_model_instances]
        for instance_id in delete_targets:
            try:
                request_json(f"{base_url}/instance/{instance_id}", method="DELETE")
                print(f"Deleted instance {instance_id}")
            except urllib.error.HTTPError as exc:
                print(f"Warning: failed to delete {instance_id}: {exc}", file=sys.stderr)
        if delete_targets:
            wait_until_gone(
                base_url,
                delete_targets,
                deadline=time.time() + 30,
                poll_interval=args.poll_interval,
            )

    create_resp = request_json(
        f"{base_url}/instance",
        method="POST",
        payload={"instance": chosen_wrapper},
    )
    print(f"Create response: {json.dumps(create_resp)}")

    deadline = time.time() + args.wait_seconds
    last_status = None
    while time.time() < deadline:
        state = fetch_json(state_url)
        active_instance = get_instance_by_id(state, chosen_instance_id)
        if active_instance is not None:
            statuses = {
                runner_id: runner_state_name(state.get("runners", {}), runner_id)
                for runner_id in runner_ids_for_instance(active_instance)
            }
            if statuses != last_status:
                print(f"Runner states: {statuses}")
                last_status = statuses
            failures = any_runner_failed(state, active_instance)
            if failures:
                print(
                    f"ERROR: runner failure detected for {chosen_instance_id}: {json.dumps(failures)}",
                    file=sys.stderr,
                )
                return 4
            if all_runners_ready(state, active_instance):
                print(f"Instance is ready: {chosen_instance_id} ({chosen_kind})")
                break
        time.sleep(args.poll_interval)
    else:
        print(
            f"ERROR: timed out waiting for instance {chosen_instance_id} to become ready.",
            file=sys.stderr,
        )
        return 5

    if args.test:
        try:
            completion = test_completion(
                base_url, model_id, args.test_prompt, args.test_timeout
            )
            print(json.dumps(completion))
        except Exception as exc:
            print(f"ERROR: completion test failed: {exc}", file=sys.stderr)
            return 6

    return 0


if __name__ == "__main__":
    sys.exit(main())
