import torch
import logging
# from keys import mykey
import torch
import logging
import numpy as np
# from keys import mykey

# A dictionary to cache models and tokenizers to avoid reloading
import os

# Keep env vars as strings; empty means unset for our local-only stack.
os.environ.setdefault("AZURE_OPENAI_ENDPOINT", "")
os.environ.setdefault("AZURE_OPENAI_API_KEY", "")
os.environ.setdefault("XTY_API_KEY", "")

# OpenAI-compatible free / freemium providers (see mnfst/awesome-free-llm-apis).
# These do NOT ship keys in that repo — only signup links. Some need no key.
API_PROVIDERS = {
    "openai": {
        "base_url": None,
        "api_key_env": ("OPENAI_API_KEY",),
        "default_headers": None,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": ("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        "default_headers": {
            "HTTP-Referer": "https://github.com/harshalDharpure/llm_abestation",
            "X-Title": "KnowGuard",
        },
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "api_key_env": ("NVIDIA_API_KEY", "NGC_API_KEY"),
        "default_headers": None,
    },
    "kilo": {
        # No API key required for free pool (200 req/hr/IP).
        "base_url": "https://api.kilo.ai/api/gateway",
        "api_key_env": ("KILO_API_KEY",),
        "allow_no_key": True,
        "default_headers": None,
    },
    "llm7": {
        # Anonymous turbo models; optional free token raises limits.
        "base_url": "https://api.llm7.io/v1",
        "api_key_env": ("LLM7_API_KEY",),
        "allow_no_key": True,
        "default_headers": None,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": ("GROQ_API_KEY",),
        "default_headers": None,
    },
    "google": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        "default_headers": None,
    },
}

# Treat these use_api values as OpenAI-compatible chat clients.
OPENAI_COMPAT_APIS = set(API_PROVIDERS.keys()) | {"azureopenai"}

global models
models = {}


def _resolve_api_key(provider_name: str):
    cfg = API_PROVIDERS.get(provider_name, {})
    for env_name in cfg.get("api_key_env", ()):
        val = (os.getenv(env_name) or "").strip()
        if val:
            return val
    if cfg.get("allow_no_key"):
        return "no-key"
    return None


def _throttle_provider(provider_name: str):
    """Cross-process spacing for shared API quotas (esp. NVIDIA NIM 429s).

    Set KNOWGUARD_NIM_MIN_INTERVAL (seconds, default 2.5) to tune NVIDIA spacing.
    Set KNOWGUARD_API_MIN_INTERVAL for a global floor on all OpenAI-compat providers.
    """
    import time
    import fcntl

    try:
        floor = float(os.getenv("KNOWGUARD_API_MIN_INTERVAL", "0") or 0)
    except ValueError:
        floor = 0.0
    try:
        nim_gap = float(os.getenv("KNOWGUARD_NIM_MIN_INTERVAL", "2.5") or 2.5)
    except ValueError:
        nim_gap = 2.5
    gap = floor
    if provider_name == "nvidia":
        gap = max(gap, nim_gap)
    if gap <= 0:
        return

    lock_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "logs",
        f".api_throttle_{provider_name}.lock",
    )
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    stamp_path = lock_path + ".ts"
    with open(lock_path, "a+") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            last = 0.0
            if os.path.isfile(stamp_path):
                try:
                    last = float(open(stamp_path).read().strip() or 0)
                except Exception:
                    last = 0.0
            wait = last + gap - time.time()
            if wait > 0:
                time.sleep(wait)
            with open(stamp_path, "w") as sf:
                sf.write(str(time.time()))
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _load_dotenv_if_present():
    """Optional local secrets: KnowGuard/.env (gitignored)."""
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    try:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip("'").strip('"')
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv_if_present()

def log_info(message, logger_name="message_logger", print_to_std=False, mode="info"):
    logger = logging.getLogger(logger_name)
    if logger: 
        if mode == "error": logger.error(message)
        if mode == "warning": logger.warning(message)
        else: logger.info(message)
    if print_to_std: print(message + "\n")

class ModelCache:
    def __init__(self, model_name, use_vllm=False, use_api=None, **kwargs):
        self.model_name = model_name
        self.use_vllm = use_vllm
        self.use_api = use_api
        self.model = None
        self.tokenizer = None
        self.terminators = None
        self.client = None
        self.args = kwargs
        self.load_model_and_tokenizer()
    
    def load_model_and_tokenizer(self):
        if self.use_api in API_PROVIDERS:
            from openai import OpenAI
            self.api_account = self.args.get("api_account", self.use_api)
            api_key = None
            try:
                from keys import mykey
                api_key = mykey.get(self.api_account) if isinstance(mykey, dict) else None
            except Exception:
                api_key = None
            if not api_key:
                api_key = _resolve_api_key(self.use_api)
            if not api_key:
                raise RuntimeError(
                    f"No API key for use_api={self.use_api}. "
                    f"Set one of {API_PROVIDERS[self.use_api].get('api_key_env')} "
                    f"or use --use_api kilo (no key)."
                )
            cfg = API_PROVIDERS[self.use_api]
            client_kwargs = {"api_key": api_key}
            if cfg.get("base_url"):
                client_kwargs["base_url"] = self.args.get("api_base_url") or cfg["base_url"]
            if cfg.get("default_headers"):
                client_kwargs["default_headers"] = cfg["default_headers"]
            self.client = OpenAI(**client_kwargs)
        elif self.use_api == 'azureopenai':
            from openai import AzureOpenAI
            self.client = AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version="XX"
            )
        elif self.use_vllm:
            try:
                from vllm import LLM
                enable_prefix_caching = self.args.get("enable_prefix_caching", False)
                self.model = LLM(model=self.model_name, enable_prefix_caching=enable_prefix_caching)
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            except Exception as e:
                log_info(f"[ERROR] [{self.model_name}]: If using a custom local model, it is not compatible with VLLM, will load using Huggingfcae and you can ignore this error: {str(e)}", mode="error")
                self.use_vllm = False
        if not self.use_vllm and self.use_api not in OPENAI_COMPAT_APIS:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = self.args.get("device", "auto")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            load_kwargs = dict(
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                trust_remote_code=True,
            )
            if device == "auto":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    **load_kwargs,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **load_kwargs,
                ).to(device)
            self.model.eval()
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            try:
                eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                if eot_id is not None:
                    self.terminators = [self.tokenizer.eos_token_id, eot_id]
                else:
                    self.terminators = [self.tokenizer.eos_token_id]
            except:
                self.terminators = [self.tokenizer.eos_token_id]
    
    def generate(self, messages):

        self.temperature = self.args.get("temperature", 0.6)
        self.max_tokens = self.args.get("max_tokens", 256)
        self.top_p = self.args.get("top_p", 0.9)
        self.top_logprobs = self.args.get("top_logprobs", 0)

        if self.use_api in OPENAI_COMPAT_APIS:
            return self.openai_generate(messages)
        elif self.use_vllm: return self.vllm_generate(messages)
        else: return self.huggingface_generate(messages)
    
    def huggingface_generate(self, messages):
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception as e:
            log_info(f"[{self.model_name}]: Could not apply chat template to messages: {str(e)}", mode="warning")
            parts = []
            for m in messages:
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        item.get("text", "") for item in content if isinstance(item, dict)
                    )
                parts.append(str(content))
            prompt = "\n\n".join(parts)

        tokenized = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = tokenized["input_ids"]
        if hasattr(self.model, "device"):
            inputs = inputs.to(self.model.device)
        else:
            inputs = inputs.to(next(self.model.parameters()).device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                do_sample=True,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.terminators,
            )

        input_length = inputs.shape[-1]
        response_text = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        usage = {"input_tokens": input_length, "output_tokens": outputs.shape[-1] - input_length}
        return response_text, None, usage
        
        
    def vllm_generate(self, messages):
        try:
            inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        except:
            # Join messages into a single prompt for general language models
            log_info(f"[{self.model_name}]: Could not apply chat template to messages.", mode="warning")
            inputs = "\n\n".join([m['content'] for m in messages])
            # inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        from vllm import SamplingParams
        frequency_penalty = self.args.get("frequency_penalty", 0)
        presence_penalty = self.args.get("presense_penalty", 0)
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_tokens, top_p=self.top_p, logprobs=self.top_logprobs, 
                                        frequency_penalty=frequency_penalty, presence_penalty=presence_penalty)
        
        outputs = self.model.generate(inputs, sampling_params)
        response_text = outputs[0].outputs[0].text
        logprobs = outputs[0].outputs[0].cumulative_logprob
        # TODO: If top_logprobs > 0, return logprobs of generation
        # if self.top_logprobs > 0: logprobs = outputs[0].outputs[0].logprobs
        usage = {"input_tokens": len(outputs[0].prompt_token_ids), "output_tokens": len(outputs[0].outputs[0].token_ids)}
        output_dict = {'response_text': response_text, 'usage': usage}

        # log_info(f"[{self.model_name}][OUTPUT]: {output_dict}")
        return response_text, logprobs, usage

    def openai_generate(self, messages):
        import time
        # o1/o3-style models reject some sampling args; degrade gracefully.
        name = (self.model_name or "").lower()
        is_reasoning = any(x in name for x in ("o1", "o3"))
        create_kwargs = {
            "model": self.model_name,
            "messages": messages,
        }
        if not is_reasoning:
            create_kwargs["temperature"] = self.temperature
            create_kwargs["max_tokens"] = self.max_tokens
            create_kwargs["top_p"] = self.top_p
        else:
            create_kwargs["max_completion_tokens"] = self.max_tokens
        if self.top_logprobs > 0 and not is_reasoning:
            create_kwargs["logprobs"] = True
            create_kwargs["top_logprobs"] = self.top_logprobs

        last_err = None
        response = None
        # NIM/shared quotas often 429 under multi-shard load; prefer long backoff
        # over crashing the worker after a short retry budget.
        max_attempts = 24
        for attempt in range(max_attempts):
            try:
                _throttle_provider(self.use_api or "")
                response = self.client.chat.completions.create(**create_kwargs)
                break
            except Exception as e:
                last_err = e
                msg = str(e).lower()
                is_rate = any(x in msg for x in ("429", "rate", "too many requests"))
                retryable = is_rate or any(
                    x in msg
                    for x in ("timeout", "503", "502", "overloaded", "capacity", "unavailable")
                )
                if not retryable or attempt == max_attempts - 1:
                    raise
                # Rate limits: slower exponential (cap 5 min). Other transient: prior curve.
                if is_rate:
                    sleep_s = min(300, (3 ** min(attempt, 5)) + 5)
                else:
                    sleep_s = min(120, (2 ** attempt) + 1)
                log_info(
                    f"[{self.model_name}] API retry {attempt+1}/{max_attempts} after {sleep_s}s: {e}",
                    mode="warning",
                )
                time.sleep(sleep_s)
        if response is None:
            raise last_err

        usage = getattr(response, "usage", None)
        num_input_tokens = getattr(usage, "prompt_tokens", 0) or 0
        num_output_tokens = getattr(usage, "completion_tokens", 0) or 0
        choice = response.choices[0]
        msg = choice.message
        response_text = msg.content
        if response_text is not None:
            response_text = response_text.strip()
        else:
            response_text = ""
        # Some free reasoning models put text only in `reasoning` when max_tokens is tight.
        if not response_text:
            for attr in ("reasoning", "reasoning_content"):
                alt = getattr(msg, attr, None)
                if isinstance(alt, str) and alt.strip():
                    response_text = alt.strip()
                    break
        log_probs = None
        if self.top_logprobs > 0 and getattr(choice, "logprobs", None):
            log_probs = choice.logprobs.top_logprobs
        return response_text, log_probs, {"input_tokens": num_input_tokens, "output_tokens": num_output_tokens}


def get_response(messages, model_name, use_vllm=False, use_api=None, **kwargs):
    # Auto-route OpenAI-compatible providers from model name / env.
    if use_api is None:
        # Honor process-level preference (e.g. Ultra-only runners set USE_API=nvidia).
        env_api = (os.getenv("USE_API") or "").strip().lower()
        if env_api in API_PROVIDERS:
            use_api = env_api
    if use_api is None:
        name = (model_name or "").lower()
        # NVIDIA-hosted model ids must prefer NIM, not OpenRouter.
        if name.startswith("nvidia/") or "nemotron" in name or name.startswith("meta/"):
            if _resolve_api_key("nvidia"):
                use_api = "nvidia"
            elif _resolve_api_key("openrouter"):
                use_api = "openrouter"
            else:
                use_api = "kilo"
        elif ":free" in name or name.startswith(("openrouter/", "meta-llama/", "google/gemma", "qwen/", "minimax/")):
            if _resolve_api_key("openrouter"):
                use_api = "openrouter"
            else:
                use_api = "kilo"  # no-key free pool
        elif any(x in name for x in ("gpt-4", "gpt-3.5", "gpt-5", "o1", "o3", "chatgpt")):
            if os.getenv("OPENAI_API_KEY"):
                use_api = "openai"
            elif _resolve_api_key("openrouter"):
                use_api = "openrouter"
    model_cache = models.get(model_name, None)
    # Rebuild cache if API mode changed for same name
    if model_cache is not None and getattr(model_cache, "use_api", None) != use_api:
        model_cache = None
        models.pop(model_name, None)
    if model_cache is None:
        model_cache = ModelCache(model_name, use_vllm=use_vllm, use_api=use_api, **kwargs)
        models[model_name] = model_cache
    return model_cache.generate(messages)





class EmbeddingModel:

    def __init__(self, model_name, use_api=None, chunk_size=1000, **kwargs):
        self.model_name = model_name
        self.use_api = use_api
        self.chunk_size = chunk_size
        self.model = None
        self.tokenizer = None
        self.client = None
        self.args = kwargs
        self.load_model()
    
    def load_model(self):
        # Try to load with sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model_type = "sentence_transformer"
            log_info(f"[{self.model_name}]: Loaded embedding model using sentence-transformers")
        except ImportError:
            log_info(f"[{self.model_name}]: sentence-transformers not available, using transformers", mode="warning")
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.eval()
            self.model_type = "transformers"
            log_info(f"[{self.model_name}]: Loaded embedding model using transformers")
    
    def embed_query(self, text):
        """
        Embed a single query text
        Args:
            text: String to embed
        Returns:
            List of floats representing the embedding
        """
        return self._embed_texts([text])[0]
    
    def embed_documents(self, texts):
        """
        Embed multiple documents
        Args:
            texts: List of strings to embed
        Returns:
            List of embeddings (each embedding is a list of floats)
        """
        return self._embed_texts(texts)
    
    def _embed_texts(self, texts):
        """Internal method to embed texts"""
        if self.use_api == "openai":
            return self._openai_embeddings(texts)
        elif self.model_type == "sentence_transformer":
            return self._sentence_transformer_embeddings(texts)
        else:  # transformers
            return self._transformers_embeddings(texts)

    def _sentence_transformer_embeddings(self, texts):
        """Get embeddings using sentence-transformers"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            try:
                embeddings = self.model.encode(chunk, convert_to_numpy=True)
                all_embeddings.extend(embeddings.tolist())
            except Exception as e:
                log_info(f"[ERROR] Sentence transformer embedding failed: {str(e)}", mode="error")
                raise e
        
        return all_embeddings
    
    def _transformers_embeddings(self, texts):
        """Get embeddings using transformers AutoModel"""
        import torch
        from torch.nn.functional import normalize
        
        all_embeddings = []
        
        for i in range(0, len(texts), self.chunk_size):
            chunk = texts[i:i + self.chunk_size]
            try:
                # Tokenize
                inputs = self.tokenizer(chunk, padding=True, truncation=True, 
                                      return_tensors="pt", max_length=512)
                
                # Move to device if model has device
                if hasattr(self.model, 'device'):
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    token_embeddings = outputs.last_hidden_state
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    
                    # Normalize embeddings
                    embeddings = normalize(embeddings, p=2, dim=1)
                    
                    all_embeddings.extend(embeddings.cpu().numpy().tolist())
                    
            except Exception as e:
                log_info(f"[ERROR] Transformers embedding failed: {str(e)}", mode="error")
                raise e
        
        return all_embeddings


def get_embeddings(model_name, use_api=None, chunk_size=1000, **kwargs):
    """
    Get an embedding model wrapper similar to AzureOpenAIEmbeddings
    Args:
        model_name: Name of the embedding model
        use_api: API to use ("openai" or None for local models)
        chunk_size: Batch size for processing
    Returns:
        EmbeddingModel instance with embed_query and embed_documents methods
    """
    # Determine API usage for embedding models
    if use_api == 'openai':
        pass
    elif use_api == 'azureopenai':
        from openai import AzureOpenAI
        from langchain.embeddings import AzureOpenAIEmbeddings  # Import AzureOpenAIEmbeddings
        client = AzureOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version="2024-10-21"
        )
        embedding_model = AzureOpenAIEmbeddings(
            client=client,
            chunk_size=1000,
            azure_deployment="text-embedding-3-small"
        )
        print(embedding_model)
    else:
        # Create a unique cache key for embedding models
        embedding_model_key = f"{model_name}_embedding"
        
        embedding_model = models.get(embedding_model_key, None)
        if embedding_model is None:
            embedding_model = EmbeddingModel(
                model_name=model_name,
                use_api=use_api,
                chunk_size=chunk_size,
                **kwargs
            )
            models[embedding_model_key] = embedding_model
    
    return embedding_model
