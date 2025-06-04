# Standard
import argparse
import asyncio
import json
import logging
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional

# Third Party
import openai
import pandas as pd
import zmq
from utils import AsyncLoopWrapper, init_logger

logger = init_logger(__name__, logging.INFO)


@dataclass
class WorkloadConfig:
    # Max number of users in the system concurrently
    num_users: int

    # Length of shared system prompt
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Length of the answer in one round
    max_answer_len: int
    min_answer_len: int

    # Number of rounds in the conversation
    num_rounds: int

    # Whether to use full dialogue without capping to num_rounds
    use_full_dialogue: bool

    # Overall QPS
    qps: int

    # Model name
    model: str

    # Whether to include user id in request header
    enable_user_id: bool


@dataclass
class UserConfig:
    # User id
    user_id: int

    # System prompt length
    system_prompt_len: int

    # Length of the user-specific data
    user_info_len: int

    # Answer length
    max_answer_len: int
    min_answer_len: int

    # Gap between two requests
    gap_between_requests: int

    # Num rounds
    num_rounds: int

    # Whether to use full dialogue without capping to num_rounds
    use_full_dialogue: bool

    # Whether to include user id in request header
    enable_user_id: bool

    @staticmethod
    def new_user_config(user_id: int, workload_config: WorkloadConfig) -> "UserConfig":
        return UserConfig(
            user_id=user_id,
            system_prompt_len=workload_config.system_prompt_len,
            user_info_len=workload_config.user_info_len,
            max_answer_len=workload_config.max_answer_len,
            min_answer_len=workload_config.min_answer_len,
            gap_between_requests=1 / workload_config.qps,
            num_rounds=workload_config.num_rounds,
            use_full_dialogue=workload_config.use_full_dialogue,
            enable_user_id=workload_config.enable_user_id,
        )


class ChatHistory:
    def __init__(
        self,
    ):
        self.history = []

    def on_user_query(self, query: str):
        if len(self.history) == 0:
            self.history.append({"role": "user", "content": query})
        else:
            assert self.history[-1]["role"] == "assistant", "Expect system response"
            self.history.append({"role": "user", "content": query})

    def on_system_response(self, response: str):
        assert len(self.history) > 0, "Expect user query"
        assert self.history[-1]["role"] == "user", "Expect user query"
        self.history.append({"role": "assistant", "content": response})

    def get_messages_for_openai(self):
        return self.history

    def __len__(self):
        return len(self.history)


@dataclass
class Response:
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


class RequestExecutor:
    def __init__(self, base_url: str, api_key: str, model: str):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.request_history = []

    async def _async_launch_request(self, messages, max_tokens, min_tokens, extra_headers=None):
        start_time = time.time()
        first_token_time = None
        words = ""

        response = await self.client.chat.completions.create(
            messages=messages,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            extra_headers=extra_headers,
            extra_body={"min_tokens": min_tokens} if min_tokens is not None else None
        )

        async for tok in response:
            if not tok.choices:
                continue
            chunk_message = tok.choices[0].delta.content
            if chunk_message is not None:
                if first_token_time is None and chunk_message != "":
                    first_token_time = time.time()
                words += chunk_message
        tokens_out = tok.usage.completion_tokens
        tokens_prefill = tok.usage.prompt_tokens

        return Response(
            body=words,
            ttft=first_token_time - start_time,
            generation_time=time.time() - first_token_time,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=time.time(),
        )

    def launch_request(
        self,
        chat_history: ChatHistory,
        max_tokens: int,
        min_tokens: int,
        finish_callback,
        extra_headers=None,
    ):
        """
        finish_callback: Callable[[Response], None]
        """
        messages = chat_history.get_messages_for_openai()
        real_callback = lambda x: finish_callback(x.result())
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(messages, max_tokens, min_tokens, extra_headers),
            self.loop,
        )
        future.add_done_callback(real_callback)


class UserSession:
    def __init__(self, user_config: UserConfig, use_sharegpt=False, sharegpt_data=None):
        self.user_config = user_config
        self.last_request_time = None
        self.chat_history = ChatHistory()
        self.question_id = 0
        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self.sharegpt_data = sharegpt_data
            if self.sharegpt_data["num_round"] % 2 == 0:
                self.start_with_gpt = False
            else:
                self.start_with_gpt = True

        self.has_unfinished_request = False
        self.last_unfinished_log = 0

        self.prompt_lengths = []
        self.generation_lengths = []
        self.ttfts = []
        self.generation_times = []
        self.launch_times = []
        self.finish_times = []

        self.finished = False

    def _update_result(self, response: Response):
        self.prompt_lengths.append(response.prompt_tokens)
        self.generation_lengths.append(response.generation_tokens)
        self.ttfts.append(response.ttft)
        self.generation_times.append(response.generation_time)
        self.launch_times.append(response.launch_time)
        self.finish_times.append(response.finish_time)

    def _build_system_prompt(self):
        def gen_dummy_text(length):
            return " ".join(["hi"] * length)

        dummy_text_sys = gen_dummy_text(self.user_config.system_prompt_len)
        dummy_text_user = gen_dummy_text(self.user_config.user_info_len)
        system_prompt = (
            f"Hi, here's some system prompt: {dummy_text_sys}."
            + f"For user {self.user_config.user_id}, "
            + f"here are some other context: {dummy_text_user}."
        )
        return system_prompt

    def _build_new_question(self):
        self.question_id += 1
        return f"Here's question #{self.question_id}: can you tell me " + "a new long story with a happy ending?"

    def _prepare_start_with_gpt(self, gpt_text: str, human_text: str) -> str:
        return f"Here's the conversation history: {gpt_text}. Here's the new question: {human_text}"

    def _launch_new_request(self, timestamp: float, request_executor: RequestExecutor):
        if self.use_sharegpt:
            if self.start_with_gpt:
                prompt = self._prepare_start_with_gpt(self.sharegpt_data["conversations"][2 * self.question_id ]["value"], self.sharegpt_data["conversations"][2 * self.question_id + 1]["value"])
            else:
                prompt = self.sharegpt_data["conversations"][2 * self.question_id]["value"]
            self.question_id += 1
        else:
            prompt = self._build_new_question()
        if len(self.chat_history) == 0:
            prompt = self._build_system_prompt() + prompt
        self.chat_history.on_user_query(prompt)
        logger.debug(f"User {self.user_config.user_id} issues request {self.question_id}")
        if self.use_sharegpt:
            if self.start_with_gpt:
                question_id = 2 * self.question_id
            else:
                question_id = 2 * self.question_id - 1
            if "num_tokens" in self.sharegpt_data["conversations"][question_id]:
                max_tokens = min(self.sharegpt_data["conversations"][question_id]["num_tokens"], self.user_config.max_answer_len)
            else:
                max_tokens = self.user_config.max_answer_len
        else:
            max_tokens = self.user_config.max_answer_len
        max_tokens = max(max_tokens, self.user_config.min_answer_len)

        request_executor.launch_request(
            self.chat_history,
            max_tokens,
            self.user_config.min_answer_len,
            self._on_request_finished,
            extra_headers={"x-user-id": str(self.user_config.user_id)},
        )
        self.has_unfinished_request = True
        self.last_request_time = timestamp

    def _on_request_finished(self, response: Response):
        self.chat_history.on_system_response(response.body)
        self.has_unfinished_request = False
        logger.debug(
            f"User {self.user_config.user_id} finished one request. "
            f"Prompt tokens: {response.prompt_tokens}, "
            f"generation tokens: {response.generation_tokens}"
        )
        self._update_result(response)

    def set_internal_state(self, offset: float, timestamp: float):
        """Tell the session is the 'offset' seconds after the start"""
        assert len(self.chat_history) == 0, "Internal state should be set before the first request"

        num_passed_questions = int(offset / self.user_config.gap_between_requests) + 1

        passed_time = (num_passed_questions - 1) * self.user_config.gap_between_requests

        self.last_request_time = timestamp - offset + passed_time
        self.question_id = num_passed_questions
        logger.debug(
            f"Set internal state for user {self.user_config.user_id}, "
            f"question_id: {self.question_id}, "
            f"last_request_time: {self.last_request_time}"
        )

    def step(self, timestamp: float, request_executor: RequestExecutor):
        if self.user_config.use_full_dialogue:
            if self.question_id >= self.sharegpt_data["num_round"] / 2 - 1 and not self.has_unfinished_request:
                self.finished = True
                return
        if not self.user_config.use_full_dialogue and self.question_id >= self.user_config.num_rounds and not self.has_unfinished_request:
            self.finished = True
            return

        if self.last_request_time is None:
            self._launch_new_request(timestamp, request_executor)
            return

        if timestamp - self.last_request_time > self.user_config.gap_between_requests:
            if self.has_unfinished_request:
                if timestamp - self.last_unfinished_log > 10:
                    logger.warning(
                        f"User {self.user_config.user_id} has an unfinished "
                        "request and unable to fit the QPS requirement."
                    )
                    self.last_unfinished_log = timestamp
                return

            self._launch_new_request(timestamp, request_executor)
            return

    def summary(self) -> pd.DataFrame:
        df = pd.DataFrame()
        df["prompt_tokens"] = self.prompt_lengths
        df["generation_tokens"] = self.generation_lengths
        df["ttft"] = self.ttfts
        df["generation_time"] = self.generation_times
        df["user_id"] = self.user_config.user_id
        df["question_id"] = range(1, len(self.prompt_lengths) + 1)
        df["launch_time"] = self.launch_times
        df["finish_time"] = self.finish_times
        return df


class UserSessionManager:
    def __init__(
        self,
        workload_config: WorkloadConfig,
        init_user_id=0,
        use_sharegpt=False,
    ):
        self.workload_config = workload_config
        self.sessions = []

        self.gap_between_requests = 1 / workload_config.qps
        self.ramp_up_time = workload_config.num_users * self.gap_between_requests
        logger.info(f"Using {self.gap_between_requests} secs between requests.")

        self.user_id = init_user_id
        self.last_user_join = 0
        self.session_summaries = []
        self.start_time = None

        self.need_ramp_up = True

        self.use_sharegpt = use_sharegpt
        if self.use_sharegpt:
            self._load_sharegpt_data()

        self.hot_cache_hit_tokens = 0
        self.backend_hit_tokens = 0
        self.local_cache_usage = 0
        self.remote_cache_usage = 0
        self.local_storage_usage = 0
        self.num_requested_tokens = 0

    def update_cache_stats(
        self,
        hot_cache_hit_tokens: int,
        backend_hit_tokens: int,
        local_cache_usage: int,
        remote_cache_usage: int,
        local_storage_usage: int,
        num_requested_tokens: int,
    ):
        self.hot_cache_hit_tokens = hot_cache_hit_tokens
        self.backend_hit_tokens = backend_hit_tokens
        self.local_cache_usage = local_cache_usage
        self.remote_cache_usage = remote_cache_usage
        self.local_storage_usage = local_storage_usage
        self.num_requested_tokens = num_requested_tokens

    def _load_sharegpt_data(self):
        with open("ShareGPT.json", "r", encoding="utf-8") as file:
            self.sharegpt_data = json.load(file)
        max_rounds = max([d["num_round"] for d in self.sharegpt_data])
        logger.info(f"There are {len(self.sharegpt_data)} users in the ShareGPT dataset. Exporting distribution of rounds to sharegpt_round_distribution.png")
        plt.hist([d["num_round"] for d in self.sharegpt_data], bins=range(0, max_rounds + 10, 10), edgecolor="black")
        plt.savefig("sharegpt_round_distribution.png")
        plt.close()

        self.sharegpt_data = [d for d in self.sharegpt_data if d["num_round"] > 2 * self.workload_config.num_rounds]
        logger.info(f"After filtering, there are {len(self.sharegpt_data)} users in the ShareGPT dataset. Exporting distribution of rounds to sharegpt_round_distribution_filtered.png.")
        if not self.workload_config.use_full_dialogue:
            logger.warning("Without use_full_dialogue, only num_rounds turns will be used.")
        plt.hist([d["num_round"] for d in self.sharegpt_data], bins=range(max(0, self.workload_config.num_rounds - 10), max_rounds + 10, 10), edgecolor="black")
        plt.savefig("sharegpt_round_distribution_filtered.png")
        plt.close()

        logger.info(f"There are {len(self.sharegpt_data)} users satisfying ")

    def _ramp_up(self, timestamp: float, ramp_up_time: float):
        for i in range(self.workload_config.num_users):
            new_session = self._create_user_session()
            offset = ramp_up_time - i * self.gap_between_requests
            if offset < 0:
                break
            new_session.set_internal_state(offset, timestamp)
        self.need_ramp_up = False

    def _create_user_session(self):
        self.user_id += 1
        user_config = UserConfig.new_user_config(self.user_id, self.workload_config)
        if self.use_sharegpt:
            user_session = UserSession(user_config, self.use_sharegpt, self.sharegpt_data[self.user_id])
        else:
            user_session = UserSession(user_config, self.use_sharegpt)
        self.sessions.append(user_session)
        return user_session

    def _remove_finished_sessions(self):
        sessions_to_remove = [s for s in self.sessions if s.finished]
        if len(sessions_to_remove) > 0:
            logger.info(
                f"Removing {len(sessions_to_remove)} finished sessions, now "
                f"active users: {len(self.sessions) - len(sessions_to_remove)}"
            )
            for session in sessions_to_remove:
                self.session_summaries.append(session.summary())
        self.sessions = [s for s in self.sessions if not s.finished]

    def step(self, timestamp: float, executor: RequestExecutor):
        if self.need_ramp_up:
            self._ramp_up(timestamp, self.ramp_up_time)

        if self.start_time is None:
            self.start_time = timestamp

        if len(self.sessions) < self.workload_config.num_users:
            self._create_user_session()
            self.last_user_join = timestamp
            logger.info(f"Joined a new user {self.user_id}, " f"now active users: {len(self.sessions)}")

        for session in self.sessions:
            session.step(timestamp, executor)

        self._remove_finished_sessions()

    @staticmethod
    def ProcessSummary(
        df: pd.DataFrame,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        pending_queries: int = 0,
        qps: Optional[int] = None,
        hot_cache_hit_tokens: int = 0,
        backend_hit_tokens: int = 0,
        local_cache_usage: int = 0,
        remote_cache_usage: int = 0,
        local_storage_usage: int = 0,
        num_requested_tokens: int = 0,
    ):
        if start_time and end_time:
            launched_queries = len(df.query(f"{start_time} <= launch_time <= {end_time}"))
            df = df.query(f"{start_time} <= finish_time <= {end_time}")
        else:
            launched_queries = len(df)

        logger.debug(
            f"Launched queries: {launched_queries}, "
            f"pending queries: {pending_queries}, "
            f"finished queries: {len(df)}"
        )

        if qps is None:
            qps = 0.0

        if start_time is None:
            start_time = df["launch_time"].min()
        if end_time is None:
            end_time = df["finish_time"].max()
        total_time = end_time - start_time

        total_requests = launched_queries + pending_queries
        _qps = total_requests / total_time

        total_finished_requests = len(df)
        finished_qps = total_finished_requests / total_time

        total_prompt_tokens = df["prompt_tokens"].sum()
        total_generation_tokens = df["generation_tokens"].sum()
        average_prefill_speed = total_prompt_tokens / total_time
        average_generation_speed = total_generation_tokens / total_time
        average_generation_speed_per_request = (df["generation_tokens"] / df["generation_time"]).mean()
        average_ttft = df["ttft"].mean()
        logger.info("Calculating performance summary")
        print("\n")
        print("==================== Performance summary ======================")
        print(f"  Launched queries: {launched_queries}")
        print(f"  Pending queries: {pending_queries}")
        print(f"  Finished queries: {len(df)}\n")
        print(f"  Total prompt tokens: {total_prompt_tokens}")
        print(f"  Total generation tokens: {total_generation_tokens}\n")
        print(f"  Local cache usage: {local_cache_usage}")
        print(f"  Remote cache usage: {remote_cache_usage}")
        print(f"  Local storage usage: {local_storage_usage}\n")
        print(f"  Hot cache hit tokens: {hot_cache_hit_tokens}")
        print(
            f"  Hot cache hit rate: {0 if num_requested_tokens == 0 else hot_cache_hit_tokens / num_requested_tokens * 100:.2f}%"
        )
        print(f"  Backend hit tokens: {backend_hit_tokens}")
        print(
            f"  Backend hit rate: {0 if num_requested_tokens == 0 else backend_hit_tokens / num_requested_tokens * 100:.2f}%\n"
        )
        print(f"  QPS: {qps:.4f} reqs/s/user\n")
        print(f"  Processing speed: {finished_qps:.4f} reqs/s\n")
        print(f"  Requests on-the-fly: {pending_queries}\n")
        print(f"  Input tokens per second: {average_prefill_speed:.4f} tokens/s\n")
        print(f"  Output tokens per second: {average_generation_speed:.4f} tokens/s\n")
        print(
            f"  Average generation throughput (per request): {average_generation_speed_per_request:.4f} tokens/req/s\n"
        )
        print(f"  Average TTFT: {average_ttft:.4f}s\n")
        print(f"Time range: {start_time} - {end_time} ({total_time:.2f}s)")
        print("===============================================================")
        print("\n")

        summary = {
            "Launched queries": launched_queries,
            "Pending queries": pending_queries,
            "Finished queries": len(df),
            "Total prompt tokens": total_prompt_tokens,
            "Total generation tokens": total_generation_tokens,
            "Hot cache hit tokens": hot_cache_hit_tokens,
            "Hot cache hit rate (%)": 0
            if total_prompt_tokens == 0
            else hot_cache_hit_tokens / total_prompt_tokens * 100,
            "Backend hit tokens": backend_hit_tokens,
            "Backend hit rate (%)": 0
            if total_prompt_tokens == 0
            else backend_hit_tokens / total_prompt_tokens * 100,
            "QPS (requests/s)": qps,
            "Processing speed (requests/s)": finished_qps,
            "Requests on-the-fly": pending_queries,
            "Input tokens/sec": average_prefill_speed,
            "Output tokens/sec": average_generation_speed,
            "Output tokens/request/sec": average_generation_speed_per_request,
            "Average TTFT (s)": average_ttft,
            "Start time": str(start_time),
            "End time": str(end_time),
            "Total time (s)": total_time,
        }
        df_summary = pd.DataFrame([summary])

        return df, df_summary

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        if len(self.session_summaries) == 0 and len(self.sessions) == 0:
            return pd.DataFrame()

        df = pd.concat([s for s in self.session_summaries] + [s.summary() for s in self.sessions])
        pending_queries = len([s for s in self.sessions if s.has_unfinished_request])
        start_time = max(self.start_time, start_time)
        end_time = min(end_time, df["finish_time"].max())
        qps = self.workload_config.qps

        df, df_summary = UserSessionManager.ProcessSummary(
            df,
            start_time,
            end_time,
            pending_queries,
            qps,
            self.hot_cache_hit_tokens,
            self.backend_hit_tokens,
            self.local_cache_usage,
            self.remote_cache_usage,
            self.local_storage_usage,
            self.num_requested_tokens,
        )
        return df, df_summary


def warmup_engine(executor):
    logger.info("Warming up the engine")
    for i in range(10):
        chat_history = ChatHistory()
        chat_history.on_user_query(f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}.")
        executor.launch_request(chat_history, 100, None, lambda x: None)

    AsyncLoopWrapper.WaitLoop()


def parse_arguments() -> WorkloadConfig:
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.")

    parser.add_argument("--num-users", type=int, required=True, help="Max number of users in the system concurrently")
    parser.add_argument(
        "--shared-system-prompt", type=int, required=True, help="Length of the shared system prompt (tokens)"
    )
    parser.add_argument(
        "--user-history-prompt", type=int, required=True, help="Length of the user-specific history prompt (tokens)"
    )
    parser.add_argument("--max-answer-len", type=int, required=True, help="Length of the answer in one round")
    parser.add_argument("--min-answer-len", type=int, required=True, help="Minimum length of the answer in one round")
    parser.add_argument("--num-rounds", type=int, required=True, help="Number of rounds in the conversation")
    parser.add_argument("--use-full-dialogue", action="store_true", help="Use full dialogue without capping to num_rounds")
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--base-url", type=str, required=True, help="Base URL of the serving engine endpoint")
    parser.add_argument("--time", type=int, required=False, help="The time to run the simulation in seconds")
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) for the summary csv and txt",
    )
    parser.add_argument("--init-user-id", type=int, default=0, help="The initial user id to start with")
    parser.add_argument(
        "--request-with-user-id", action="store_true", help="Whether to enable user id in the request headers"
    )
    parser.add_argument("--log-interval", type=int, default=30, help="The time between two summary loggings in seconds")

    parser.add_argument("--verbose", action="store_true", help="Whether to enable verbose logging")
    parser.add_argument("--sharegpt", action="store_true", help="Whether to use ShareGPT dataset")
    args = parser.parse_args()
    return args


def parse_process_summary():
    parser = argparse.ArgumentParser(description="Parse benchmark configurations.", add_help=False)

    parser.add_argument("--process-summary", type=str, default=None)

    args, _ = parser.parse_known_args()
    return args


def process_output(filename):
    logger.warning(f"Processing the existing summary file {filename}" ", ignoring all the other arguments")
    UserSessionManager.ProcessSummary(pd.read_csv(filename), pending_queries=0)


def main():
    args = parse_process_summary()
    if args.process_summary:
        process_output(args.process_summary)
        return

    args = parse_arguments()
    if args.verbose:
        global logger
        logger = init_logger(__name__, level=logging.DEBUG)

    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 1000)
    socket.setsockopt(zmq.SNDTIMEO, 1000)
    socket.setsockopt(zmq.IMMEDIATE, 1)

    step_interval = 0.1

    executor = RequestExecutor(base_url=args.base_url, api_key="EMPTY", model=args.model)

    warmup_engine(executor)
    workload_config = WorkloadConfig(
        num_users=args.num_users,
        system_prompt_len=args.shared_system_prompt,
        user_info_len=args.user_history_prompt,
        max_answer_len=args.max_answer_len,
        min_answer_len=args.min_answer_len,
        num_rounds=args.num_rounds,
        use_full_dialogue=args.use_full_dialogue,
        qps=args.qps,
        model=args.model,
        enable_user_id=args.request_with_user_id,
    )

    manager = UserSessionManager(
        workload_config,
        init_user_id=args.init_user_id,
        use_sharegpt=args.sharegpt,
    )

    num_steps = 0
    start_time = time.time()
    last_summary_time = start_time
    try:
        while True:
            num_steps += 1
            manager.step(time.time(), executor)
            time.sleep(step_interval)

            # if time.time() - last_summary_time > args.log_interval:
            #     manager.summary(last_summary_time, time.time())
            #     last_summary_time = time.time()

            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    try:
        socket.connect(f"ipc:///tmp/lmcache_socket")
        socket.send_string("get")
        response = socket.recv_json()
        manager.update_cache_stats(
            response["data"]["hot_cache_hit_tokens"],
            response["data"]["backend_hit_tokens"],
            response["data"]["local_cache_usage"],
            response["data"]["remote_cache_usage"],
            response["data"]["local_storage_usage"],
            response["data"]["num_requested_tokens"],
        )
    except Exception as e:
        logger.warning(f"Failed to get cache stats: {e}")

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    df, df_summary = manager.summary(0, time.time())
    df.to_csv(args.output, index=False)
    df_summary.to_csv(f"{'.'.join(args.output.split('.')[:-1])}_summary.csv", index=False)


if __name__ == "__main__":
    main()
