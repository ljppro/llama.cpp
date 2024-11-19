#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# type: ignore[reportUnusedImport]

import subprocess
import os
import sys
import threading
import requests
import time
from typing import (
    Any,
    Callable,
    ContextManager,
    Iterable,
    Iterator,
    Literal,
    Sequence,
    Set,
)


class ServerResponse:
    headers: dict
    status_code: int
    body: dict


class ServerProcess:
    # default options
    debug: bool = False
    server_port: int = 8080
    server_host: str = "127.0.0.1"
    model_hf_repo: str = "ggml-org/models"
    model_hf_file: str = "tinyllamas/stories260K.gguf"

    # custom options
    model_alias: str | None = None
    model_url: str | None = None
    model_file: str | None = None
    n_threads: int | None = None
    n_gpu_layer: str | None = None
    n_batch: int | None = None
    n_ubatch: int | None = None
    n_ctx: int | None = None
    n_ga: int | None = None
    n_ga_w: int | None = None
    n_predict: int | None = None
    n_prompts: int | None = 0
    n_server_predict: int | None = None
    slot_save_path: str | None = None
    id_slot: int | None = None
    cache_prompt: bool | None = None
    n_slots: int | None = None
    server_api_key: str | None = None
    server_continuous_batching: bool | None = False
    server_embeddings: bool | None = False
    server_reranking: bool | None = False
    server_metrics: bool | None = False
    seed: int | None = None
    draft: int | None = None
    server_seed: int | None = None
    user_api_key: str | None = None
    response_format: str | None = None
    temperature: float | None = None
    lora_file: str | None = None
    disable_ctx_shift: int | None = False

    # session variables
    process: subprocess.Popen | None = None

    def __init__(self):
        pass

    def start(self, timeout_seconds: int = 10) -> None:
        if "LLAMA_SERVER_BIN_PATH" in os.environ:
            server_path = os.environ["LLAMA_SERVER_BIN_PATH"]
        elif os.name == "nt":
            server_path = "../../../build/bin/Release/llama-server.exe"
        else:
            server_path = "../../../build/bin/llama-server"
        server_args = [
            "--slots",  # requires to get slot status via /slots endpoint
            "--host",
            self.server_host,
            "--port",
            self.server_port,
        ]
        if self.model_file:
            server_args.extend(["--model", self.model_file])
        if self.model_url:
            server_args.extend(["--model-url", self.model_url])
        if self.model_hf_repo:
            server_args.extend(["--hf-repo", self.model_hf_repo])
        if self.model_hf_file:
            server_args.extend(["--hf-file", self.model_hf_file])
        if self.n_batch:
            server_args.extend(["--batch-size", self.n_batch])
        if self.n_ubatch:
            server_args.extend(["--ubatch-size", self.n_ubatch])
        if self.n_threads:
            server_args.extend(["--threads", self.n_threads])
        if self.n_gpu_layer:
            server_args.extend(["--n-gpu-layers", self.n_gpu_layer])
        if self.draft is not None:
            server_args.extend(["--draft", self.draft])
        if self.server_continuous_batching:
            server_args.append("--cont-batching")
        if self.server_embeddings:
            server_args.append("--embedding")
        if self.server_reranking:
            server_args.append("--reranking")
        if self.server_metrics:
            server_args.append("--metrics")
        if self.model_alias:
            server_args.extend(["--alias", self.model_alias])
        if self.n_ctx:
            server_args.extend(["--ctx-size", self.n_ctx])
        if self.n_slots:
            server_args.extend(["--parallel", self.n_slots])
        if self.n_server_predict:
            server_args.extend(["--n-predict", self.n_server_predict])
        if self.slot_save_path:
            server_args.extend(["--slot-save-path", self.slot_save_path])
        if self.server_api_key:
            server_args.extend(["--api-key", self.server_api_key])
        if self.n_ga:
            server_args.extend(["--grp-attn-n", self.n_ga])
        if self.n_ga_w:
            server_args.extend(["--grp-attn-w", self.n_ga_w])
        if self.debug:
            server_args.append("--verbose")
        if self.lora_file:
            server_args.extend(["--lora", self.lora_file])
        if self.disable_ctx_shift:
            server_args.extend(["--no-context-shift"])

        args = [str(arg) for arg in [server_path, *server_args]]
        print(f"bench: starting server with: {' '.join(args)}")

        flags = 0
        if "nt" == os.name:
            flags |= subprocess.DETACHED_PROCESS
            flags |= subprocess.CREATE_NEW_PROCESS_GROUP
            flags |= subprocess.CREATE_NO_WINDOW

        self.process = subprocess.Popen(
            [str(arg) for arg in [server_path, *server_args]],
            creationflags=flags,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "LLAMA_CACHE": "tmp"},
        )
        server_instances.add(self)

        def server_log(in_stream, out_stream):
            for line in iter(in_stream.readline, b""):
                print(line.decode("utf-8"), end="", file=out_stream)

        thread_stdout = threading.Thread(
            target=server_log, args=(self.process.stdout, sys.stdout), daemon=True
        )
        thread_stdout.start()

        thread_stderr = threading.Thread(
            target=server_log, args=(self.process.stderr, sys.stderr), daemon=True
        )
        thread_stderr.start()

        print(f"server pid={self.process.pid}, pytest pid={os.getpid()}")

        # wait for server to start
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                response = self.make_request("GET", "/slots")
                if response.status_code == 200:
                    self.ready = True
                    return  # server is ready
            except Exception as e:
                pass
            print(f"Waiting for server to start...")
            time.sleep(0.5)
        raise TimeoutError(f"Server did not start within {timeout_seconds} seconds")

    def stop(self) -> None:
        server_instances.remove(self)
        if self.process:
            print(f"Stopping server with pid={self.process.pid}")
            self.process.kill()
            self.process = None

    def make_request(
        self,
        method: str,
        path: str,
        data: dict | None = None,
        headers: dict | None = None,
    ) -> ServerResponse:
        url = f"http://{self.server_host}:{self.server_port}{path}"
        headers = {}
        if self.user_api_key:
            headers["Authorization"] = f"Bearer {self.user_api_key}"
        if self.response_format:
            headers["Accept"] = self.response_format
        if method == "GET":
            response = requests.get(url, headers=headers)
        elif method == "POST":
            response = requests.post(url, headers=headers, json=data)
        elif method == "OPTIONS":
            response = requests.options(url, headers=headers)
        else:
            raise ValueError(f"Unimplemented method: {method}")
        result = ServerResponse()
        result.headers = dict(response.headers)
        result.status_code = response.status_code
        result.body = response.json()
        return result


server_instances: Set[ServerProcess] = set()


def multiple_post_requests(
    server: ServerProcess, path: str, data: Sequence[dict], headers: dict | None = None
) -> Sequence[ServerResponse]:
    def worker(data_chunk):
        try:
            return server.make_request("POST", path, data=data_chunk, headers=headers)
        except Exception as e:
            print(f"Error occurred: {e}", file=sys.stderr)
            os._exit(1)  # terminate main thread

    threads = []
    results = []

    def thread_target(data_chunk):
        result = worker(data_chunk)
        results.append(result)

    for chunk in data:
        thread = threading.Thread(target=thread_target, args=(chunk,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    return results
