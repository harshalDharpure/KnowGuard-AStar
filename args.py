import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description="Run KnowGuard / MediQ-style open-ended benchmark.")

    parser.add_argument("--expert_module", type=str, default="expert")
    parser.add_argument("--expert_class", type=str, required=True)
    parser.add_argument("--expert_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument(
        "--expert_model_question_generator",
        type=str,
        default=None,
        help="Optional separate model for question generation.",
    )

    parser.add_argument("--patient_module", type=str, default="patient")
    parser.add_argument("--patient_class", type=str, default="FactSelectPatient")
    parser.add_argument("--patient_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dev_filename", type=str, required=True)
    parser.add_argument("--output_filename", type=str, default="results/results.jsonl")

    parser.add_argument("--question_type", type=str, default="open-ended", choices=["open-ended", "multiple_choice"])
    parser.add_argument("--judge_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")

    parser.add_argument("--max_questions", type=int, default=12)
    parser.add_argument("--log_filename", type=str, default="logs/run.log")
    parser.add_argument("--history_log_filename", type=str, default=None)
    parser.add_argument("--detail_log_filename", type=str, default=None)
    parser.add_argument("--message_log_filename", type=str, default=None)

    parser.add_argument("--rationale_generation", action="store_true")
    parser.add_argument("--self_consistency", type=int, default=2)
    parser.add_argument("--abstain_threshold", type=float, default=3.5)
    parser.add_argument("--kg_threshold", type=float, default=4.0)
    parser.add_argument(
        "--min_questions",
        type=int,
        default=3,
        help="Minimum patient Q&A turns before allowing answer (paper KnowGuard ~5.7 avg).",
    )
    parser.add_argument("--independent_modules", action="store_true")

    parser.add_argument("--use_vllm", action="store_true")
    parser.add_argument(
        "--use_api",
        type=str,
        default=None,
        choices=[None, "openai", "azureopenai", "openrouter", "nvidia", "kilo", "llm7", "groq", "google"],
        help="OpenAI-compatible provider. 'kilo' needs no key (free pool).",
    )
    parser.add_argument(
        "--api_base_url",
        type=str,
        default=None,
        help="Optional override for OpenAI-compatible base URL.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=768)
    parser.add_argument("--top_logprobs", type=int, default=0)
    parser.add_argument("--api_account", type=str, default="knowguard")
    parser.add_argument("--device", type=str, default="auto")

    # KnowGuard-specific
    parser.add_argument("--know_mode", type=str, default="text_only", choices=["text_only", "multimodal", "image_only"])
    parser.add_argument("--relevance_threshold", type=float, default=0.6)
    parser.add_argument("--llm_relevance_threshold", type=float, default=0.4)
    parser.add_argument("--relevance_modality", type=str, default="text")
    parser.add_argument("--initial_triplets", type=int, default=2)
    parser.add_argument("--direct_query_new", action="store_true")
    parser.add_argument("--max_queue_size", type=int, default=6)
    parser.add_argument("--embedding_weight", type=float, default=0.2)
    parser.add_argument("--llm_weight", type=float, default=0.6)
    parser.add_argument("--coherence_weight", type=float, default=0.35)
    parser.add_argument("--decay_weight", type=float, default=0.5)
    parser.add_argument("--subgraph_weight", type=float, default=1.15)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--use_question_query", action="store_true")
    parser.add_argument("--multi_hop", action="store_true", default=True)
    parser.add_argument("--max_hop_depth", type=int, default=2)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--hop_decay_factor", type=float, default=0.7)
    parser.add_argument("--use_query_generation", action="store_true", default=True)

    # Data / KG paths
    parser.add_argument("--kg_csv", type=str, default="data/kg/filtered_data_v1.csv")
    parser.add_argument("--disease2demo_csv", type=str, default="data/kg/baseline_dataset/Disease2demo.csv")
    parser.add_argument("--faiss_dir", type=str, default="data/kg/faiss_db_minilm")
    parser.add_argument("--embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--who_overview_json", type=str, default="data/kg/WHO/overview.json")
    parser.add_argument("--image_base_path", type=str, default="data/kg/WHO/")

    # Path-to-80% clinical RAG + adjudication
    parser.add_argument("--use_clinical_rag", action="store_true", help="Inject MedQA textbook passages into KnowGuard evidence")
    parser.add_argument("--clinical_corpus_dir", type=str, default="data/kg/clinical_corpus")
    parser.add_argument("--clinical_embedding_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--clinical_reranker_model", type=str, default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--clinical_rag_top_k", type=int, default=5)
    parser.add_argument("--clinical_rag_rerank", action="store_true", default=True)
    parser.add_argument("--use_adjudicator", action="store_true", help="SEMA-style sufficiency check before final answer")
    # Optional interaction / abstention flags
    parser.add_argument("--rebuild_kg_graph", action="store_true", help="Force full KG CSV scan + FAISS rebuild")
    parser.add_argument("--use_discriminative_questions", action="store_true", default=True,
                        help="Differential pruning questions (Hack 1)")
    parser.add_argument("--no_discriminative_questions", action="store_false", dest="use_discriminative_questions")
    parser.add_argument("--use_entropy_gate", action="store_true", default=True, help="Shannon entropy commit gate")
    parser.add_argument("--no_entropy_gate", action="store_false", dest="use_entropy_gate")
    parser.add_argument("--entropy_commit_threshold", type=float, default=0.9, help="Commit when H(D) <= this (nats)")
    parser.add_argument("--use_dual_process", action="store_true", default=True, help="System-2 adversarial verify")
    parser.add_argument("--no_dual_process", action="store_false", dest="use_dual_process")
    parser.add_argument("--use_council", action="store_true", default=True, help="Multi-model council at commit")
    parser.add_argument("--no_council", action="store_false", dest="use_council")
    parser.add_argument("--phase_routing", action="store_true", default=True, help="Turn-based RAG vs KG routing")
    parser.add_argument("--no_phase_routing", action="store_false", dest="phase_routing")
    parser.add_argument("--max_cases", type=int, default=None, help="Limit benchmark cases")


    args = parser.parse_args()
    if args.expert_model_question_generator is None:
        args.expert_model_question_generator = args.expert_model
    # Auto-select OpenAI-compatible provider from model name / env.
    if args.use_api is None and args.expert_model:
        low = args.expert_model.lower()
        if ":free" in low or low.startswith(("nvidia/", "openrouter/", "minimax/", "stepfun/")):
            args.use_api = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "kilo"
        elif any(x in low for x in ("gpt-4", "gpt-3.5", "gpt-5", "o1", "o3")):
            if os.getenv("OPENAI_API_KEY"):
                args.use_api = "openai"
            elif os.getenv("OPENROUTER_API_KEY"):
                args.use_api = "openrouter"
        elif low.startswith("meta/") or ("nemotron" in low and not low.endswith(":free")):
            if os.getenv("NVIDIA_API_KEY") or os.getenv("NGC_API_KEY"):
                args.use_api = "nvidia"

    for path_attr in [
        "log_filename",
        "history_log_filename",
        "detail_log_filename",
        "message_log_filename",
        "output_filename",
    ]:
        path = getattr(args, path_attr)
        if path:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
    return args