#!/usr/bin/env python3
import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request


def fetch_json(url: str):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def request_json(url: str, method: str = "GET", payload=None, timeout: int = 30):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read()
        if not body:
            return None
        return json.loads(body.decode())


def get_model_id_from_instance(instance_wrapper):
    instance_kind, inner = next(iter(instance_wrapper.items()))
    shard_assignments = inner.get("shardAssignments", {})
    return shard_assignments.get("modelId"), instance_kind, inner


def list_models(base_url):
    response = request_json(f"{base_url}/v1/models")
    data = response.get("data", []) if response else []
    return sorted(
        {
            item.get("id")
            for item in data
            if isinstance(item, dict) and item.get("id")
        }
    )


def list_instance_models(state):
    model_ids = set()
    for wrapper in state.get("instances", {}).values():
        model_id, _, _ = get_model_id_from_instance(wrapper)
        if model_id:
            model_ids.add(model_id)
    return sorted(model_ids)


def find_instances_for_model(state, model_id):
    matches = []
    for instance_id, wrapper in state.get("instances", {}).items():
        found_model_id, instance_kind, inner = get_model_id_from_instance(wrapper)
        if found_model_id == model_id:
            matches.append(
                {
                    "instance_id": instance_id,
                    "instance_kind": instance_kind,
                    "inner": inner,
                }
            )
    return matches


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
    _, _, inner = get_model_id_from_instance(instance_wrapper)
    return list(inner.get("shardAssignments", {}).get("nodeToRunner", {}).values())


def runner_state_name(runners, runner_id):
    runner = runners.get(runner_id, {})
    if not runner:
        return "Missing"
    return next(iter(runner.keys()))


def all_runners_ready(state, instance_wrapper):
    runner_ids = runner_ids_for_instance(instance_wrapper)
    runners = state.get("runners", {})
    if not runner_ids:
        return False
    readyish = {"RunnerReady", "RunnerIdle"}
    return all(runner_state_name(runners, rid) in readyish for rid in runner_ids)


def any_runner_failed(state, instance_wrapper):
    runner_ids = runner_ids_for_instance(instance_wrapper)
    runners = state.get("runners", {})
    failures = {}
    for rid in runner_ids:
        runner = runners.get(rid, {})
        state_name = next(iter(runner.keys())) if runner else "Missing"
        if state_name == "RunnerFailed":
            failures[rid] = runner[state_name]
    return failures


def test_completion(base_url, model_id, prompt, timeout):
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": 8,
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


def prompt_for_model(instance_models, api_models):
    print("Multiple models are available, so model selection is needed.")
    if instance_models:
        print("Active instance models:")
        for index, model in enumerate(instance_models, start=1):
            print(f"  {index}. {model}")
        print("Enter one of the numbers above, paste an exact model ID, or type search text.")
    else:
        print("No active instance model could be auto-selected.")
        print("Paste an exact model ID or type search text to filter /v1/models.")

    last_matches = []
    while True:
        prompt = "Model selection: "
        try:
            response = input(prompt).strip()
        except EOFError as exc:
            raise ValueError(
                "Interactive model selection requires a TTY. Pass --model <model-id> explicitly."
            ) from exc
        except KeyboardInterrupt as exc:
            raise ValueError("Model selection cancelled by user.") from exc

        if not response:
            print("Please enter a model number, exact model ID, or search text.")
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

        matches = [model for model in api_models if response.lower() in model.lower()]
        if len(matches) == 1:
            print(f"Selected: {matches[0]}")
            return matches[0]
        if not matches:
            print("No matching models found. Refine the search or paste an exact model ID.")
            continue

        last_matches = matches[:20]
        print(f"Found {len(matches)} matching models:")
        for index, model in enumerate(last_matches, start=1):
            print(f"  {index}. {model}")
        if len(matches) > len(last_matches):
            print("Showing the first 20 matches. Narrow the search or paste an exact model ID.")
        else:
            print("Enter one of the numbers above, refine the search, or paste an exact model ID.")


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


def main():
    parser = argparse.ArgumentParser(
        description="Recreate a local EXO model instance as MlxRing and optionally test the API."
    )
    parser.add_argument("--base-url", default="http://localhost:52415")
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID to reload. Omit or pass 'auto' to auto-resolve when possible.",
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
        return 4

    print(
        f"Chosen preview: meta={chosen.get('instance_meta')} sharding={chosen.get('sharding')}"
    )

    if args.dry_run:
        return 0

    if not args.keep_existing:
        if args.delete_all_instances:
            delete_targets = list(state.get("instances", {}).keys())
        else:
            delete_targets = [item["instance_id"] for item in existing_model_instances]
        for instance_id in delete_targets:
            try:
                request_json(f"{base_url}/instance/{instance_id}", method="DELETE")
                print(f"Deleted instance {instance_id}")
            except urllib.error.HTTPError as exc:
                print(f"Warning: failed to delete {instance_id}: {exc}", file=sys.stderr)

    create_resp = request_json(
        f"{base_url}/instance",
        method="POST",
        payload={"instance": chosen["instance"]},
    )
    print(f"Create response: {json.dumps(create_resp)}")

    deadline = time.time() + args.wait_seconds
    last_status = None
    active_instance = None
    while time.time() < deadline:
        state = fetch_json(state_url)
        instances = find_instances_for_model(state, model_id)
        if instances:
            active_instance = {
                next(iter(state["instances"][instances[0]["instance_id"]].keys())): instances[0]["inner"]
            }
            runner_ids = runner_ids_for_instance(active_instance)
            statuses = {
                rid: runner_state_name(state.get("runners", {}), rid) for rid in runner_ids
            }
            if statuses != last_status:
                print(f"Runner states: {statuses}")
                last_status = statuses
            failures = any_runner_failed(state, active_instance)
            if failures:
                print(f"ERROR: runner failure detected: {json.dumps(failures)}", file=sys.stderr)
                return 5
            if all_runners_ready(state, active_instance):
                print("Instance is ready.")
                break
        time.sleep(args.poll_interval)
    else:
        print("ERROR: timed out waiting for runner readiness.", file=sys.stderr)
        return 6

    if args.test:
        try:
            completion = test_completion(
                base_url, model_id, args.test_prompt, args.test_timeout
            )
            print(json.dumps(completion))
        except Exception as exc:
            print(f"ERROR: completion test failed: {exc}", file=sys.stderr)
            return 7

    return 0


if __name__ == "__main__":
    sys.exit(main())
