import asyncio
import json
import uuid
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple

from core.control_contract import (
    ControlDecision,
    ExecutionResult,
    persist_control_decision,
    persist_execution_result,
)


def initialize_pipeline_orchestrator(
    orch: Any,
    context_manager: Any = None,
    *,
    context_manager_cls: Callable[[], Any],
    thinking_layer_cls: Callable[[], Any],
    control_layer_cls: Callable[[], Any],
    output_layer_cls: Callable[[], Any],
    tool_selector_cls: Callable[[], Any],
    tone_hybrid_cls: Callable[[], Any],
    query_budget_hybrid_cls: Callable[[], Any],
    domain_router_hybrid_cls: Callable[[], Any],
    get_registry_fn: Callable[[], Any],
    task_lifecycle_manager_cls: Callable[[], Any],
    get_archive_manager_fn: Callable[[], Any],
    tool_intelligence_manager_cls: Callable[[Any], Any],
    load_tool_execution_policy_fn: Callable[[], Dict[str, Any]],
    get_master_orchestrator_fn: Callable[..., Any],
    get_hub_fn: Callable[[], Any],
    ollama_base: str,
    lock_factory: Callable[[], Any],
    log_info_fn: Callable[[str], None],
) -> None:
    orch.context = context_manager or context_manager_cls()
    orch.thinking = thinking_layer_cls()
    orch.control = control_layer_cls()
    orch.output = output_layer_cls()
    orch.tool_selector = tool_selector_cls()
    orch.tone_hybrid = tone_hybrid_cls()
    orch.query_budget = query_budget_hybrid_cls()
    orch.domain_router = domain_router_hybrid_cls()
    orch.registry = get_registry_fn()
    orch.lifecycle = task_lifecycle_manager_cls()
    orch.archive_manager = get_archive_manager_fn()
    orch.tool_intelligence = tool_intelligence_manager_cls(orch.archive_manager)
    orch.tool_execution_policy = load_tool_execution_policy_fn()

    orch.master = get_master_orchestrator_fn(pipeline_orchestrator=orch)
    if hasattr(orch.master, "set_event_sink"):
        orch.master.set_event_sink(orch._persist_master_workspace_event)

    hub = get_hub_fn()
    orch.control.set_mcp_hub(hub)
    orch.ollama_base = ollama_base
    orch._conversation_grounding_state = {}
    orch._conversation_grounding_lock = lock_factory()
    orch._conversation_container_state = {}
    orch._conversation_container_lock = lock_factory()
    orch._conversation_consistency_state = {}
    orch._conversation_consistency_lock = lock_factory()

    log_info_fn("[PipelineOrchestrator] Initialized with 3 layers + ContextManager")


def build_effective_context(
    orch: Any,
    user_text: str,
    conv_id: Optional[str],
    *,
    small_model_mode: bool,
    cleanup_payload: Optional[Dict] = None,
    include_blocks: Optional[Dict] = None,
    debug_flags: Optional[Dict] = None,
    request_cache: Optional[Dict[str, Any]] = None,
) -> tuple:
    from config import get_context_trace_dryrun

    include_cfg = {
        "compact": True,
        "system_tools": True,
        "memory_data": True,
        **(include_blocks or {}),
    }
    flags = debug_flags or {}
    trace: Dict[str, Any] = {
        "small_model_mode": bool(small_model_mode),
        "context_sources": [],
        "context_blocks": {},
        "context_chars": 0,
        "context_chars_final": 0,
        "retrieval_count": 0,
        "mode": "",
        "flags": {
            "skills_prefetch_used": bool(flags.get("skills_prefetch_used", False)),
            "skills_prefetch_mode": str(flags.get("skills_prefetch_mode", "off" if small_model_mode else "full")),
            "detection_rules_used": str(flags.get("detection_rules_used", "false")),
            "output_reinjection_risk": not small_model_mode,
        },
    }

    ctx = orch.context.get_context(
        query=user_text,
        thinking_plan=cleanup_payload or {},
        conversation_id=conv_id or "",
        small_model_mode=small_model_mode,
        request_cache=request_cache,
    )
    memory_used = ctx.memory_used

    parts: list[str] = []
    part_compact = ""
    part_system_tools = ""
    part_memory_data = ""

    def _safe_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return str(value)
        return ""

    if include_cfg.get("compact") and small_model_mode:
        csv_trigger = None
        tp = cleanup_payload or {}
        if tp.get("time_reference"):
            csv_trigger = "time_reference"
        elif tp.get("is_fact_query"):
            csv_trigger = "fact_recall"
        elif tp.get("needs_memory"):
            csv_trigger = "remember"

        compact = orch._get_compact_context(
            conv_id,
            has_tool_failure=bool(flags.get("has_tool_failure", False)),
            csv_trigger=csv_trigger,
        )
        compact_text = _safe_text(compact)
        if compact_text:
            part_compact = compact_text
            parts.append(compact_text)
            trace["context_sources"].append("compact")
            trace["context_blocks"]["compact"] = len(compact_text)
            from config import get_jit_retrieval_max, get_jit_retrieval_max_on_failure

            compact_has_failure = bool(flags.get("has_tool_failure", False))
            compact_budget = (
                get_jit_retrieval_max_on_failure() if compact_has_failure else get_jit_retrieval_max()
            )
            compact_rc = 1 + (1 if compact_budget >= 2 and (conv_id or "") != "_container_events" else 0)
            trace["retrieval_count"] = max(trace["retrieval_count"], compact_rc)

    system_tools_text = _safe_text(getattr(ctx, "system_tools", ""))
    if include_cfg.get("system_tools") and system_tools_text:
        part_system_tools = system_tools_text
        parts.append(system_tools_text)
        trace["context_sources"].append("system_tools")
        trace["context_blocks"]["system_tools"] = len(system_tools_text)

    memory_data_text = _safe_text(getattr(ctx, "memory_data", ""))
    if include_cfg.get("memory_data") and memory_data_text:
        part_memory_data = memory_data_text
        parts.append(memory_data_text)
        trace["context_sources"].append("memory_data")
        trace["context_blocks"]["memory_data"] = len(memory_data_text)
        if ctx.memory_used:
            from config import get_jit_retrieval_max, get_jit_retrieval_max_on_failure

            rc_cap = get_jit_retrieval_max_on_failure() if flags.get("has_tool_failure") else get_jit_retrieval_max()
            trace["retrieval_count"] = min(trace["retrieval_count"] + 1, rc_cap)

    text = "\n".join(_safe_text(p) for p in parts if _safe_text(p)).strip()
    trace["context_chars"] = len(text)
    trace["memory_used"] = memory_used

    if small_model_mode:
        from config import get_small_model_char_cap

        cap = get_small_model_char_cap()
        if len(text) > cap:
            dropped: list[str] = []
            try1 = "\n".join(p for p in [part_compact, part_system_tools] if p).strip()
            if len(try1) <= cap:
                text = try1
                if part_memory_data:
                    dropped.append("memory_data")
            else:
                try2 = part_compact.strip()
                if len(try2) <= cap:
                    text = try2
                    if part_memory_data:
                        dropped.append("memory_data")
                    if part_system_tools:
                        dropped.append("system_tools")
                else:
                    text = part_compact[:cap] if part_compact else ""
                    if part_memory_data:
                        dropped.append("memory_data")
                    if part_system_tools:
                        dropped.append("system_tools")
                    if not text:
                        text = "[CONTEXT BUDGET EXHAUSTED. Please restate your request briefly.]"[:cap]
            trace["context_chars"] = len(text)
            for drop in dropped:
                if drop in trace["context_sources"]:
                    trace["context_sources"].remove(drop)
                trace["context_blocks"].pop(drop, None)
            orch_log_warn = getattr(orch, "_log_warn_for_utils", None)
            if callable(orch_log_warn):
                orch_log_warn(f"[CTX] CHAR_CAP enforced: {len(text)}/{cap} chars dropped={dropped}")

    if get_context_trace_dryrun():
        legacy_parts = []
        if small_model_mode:
            legacy_compact = orch._get_compact_context(
                conv_id,
                has_tool_failure=bool(flags.get("has_tool_failure", False)),
            )
            if legacy_compact:
                legacy_parts.append(legacy_compact)
        if system_tools_text:
            legacy_parts.append(system_tools_text)
        if memory_data_text:
            legacy_parts.append(memory_data_text)
        legacy = "\n".join(_safe_text(p) for p in legacy_parts if _safe_text(p)).strip()
        orch_log_info = getattr(orch, "_log_info_for_utils", None)
        if callable(orch_log_info):
            orch_log_info(
                f"[CTX-DRYRUN] new={len(text)} old={len(legacy)} "
                f"src_new={trace['context_sources']} "
                f"diff={len(text) - len(legacy):+d}chars"
            )
        trace["context_chars_final"] = len(legacy)
        return legacy, trace

    trace["context_chars_final"] = trace["context_chars"]
    return text, trace


async def check_pending_confirmation(
    orch: Any,
    user_text: str,
    conversation_id: str,
    *,
    intent_system_available: bool,
    get_intent_store_fn: Callable[[], Any],
    get_hub_fn: Callable[[], Any],
    intent_state_cls: Any,
    core_chat_response_cls: Any,
    log_info_fn: Callable[[str], None],
    log_warn_fn: Callable[[str], None],
    log_error_fn: Callable[[str], None],
) -> Optional[Any]:
    if not intent_system_available:
        return None
    store = get_intent_store_fn()
    pending = store.get_pending_for_conversation(conversation_id)
    if not pending:
        return None

    intent = pending[-1]
    text_lower = user_text.lower().strip()
    normalized_tokens = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text_lower).split()
    first_token = normalized_tokens[0] if normalized_tokens else ""

    negative = {"nein", "no", "abbrechen", "cancel", "stop", "nee"}
    is_negative = text_lower in negative or first_token in negative
    is_positive = (
        text_lower in ["ja", "yes", "ok", "bestaetigen", "mach", "los", "ja bitte", "klar"]
        or first_token in {"ja", "yes", "ok", "bestaetigen", "mach", "los", "klar"}
    )

    if getattr(intent, "intent_type", "") == "skill_clarification" and not is_negative:
        text_lower = "ja"
        is_positive = True
        is_negative = False

    if is_positive:
        intent.confirm()
        try:
            hub = get_hub_fn()
            trace_id = orch._normalize_trace_id(f"intent:{getattr(intent, 'id', '')}:{uuid.uuid4().hex[:8]}")
            intent_user_text = str(getattr(intent, "user_text", "") or "").strip()
            if not intent_user_text:
                intent_user_text = f"Erstelle den Skill {intent.skill_name}".strip()
            complexity_raw = getattr(intent, "complexity", 5)
            try:
                complexity = int(complexity_raw)
            except Exception:
                complexity = 5
            complexity = max(1, min(10, complexity))
            log_info_fn(
                f"[Orchestrator-Intent][trace={trace_id}] Using autonomous_skill_task for: {intent_user_text[:50]}..."
            )

            task_args = {
                "user_text": orch._safe_str(intent_user_text, max_len=4000),
                "intent": orch._safe_str(intent_user_text, max_len=4000),
                "complexity": complexity,
                "allow_auto_create": True,
                "execute_after_create": True,
                "prefer_create": True,
                "_trace_id": trace_id,
            }

            if getattr(intent, "intent_type", "") == "skill_clarification":
                enriched = intent.user_text + f"\nHinweis vom User: {user_text}"
                safe_enriched = orch._safe_str(enriched, max_len=4000)
                task_args["user_text"] = safe_enriched
                task_args["intent"] = safe_enriched
                task_args["complexity"] = 3
                log_info_fn(f"[Orchestrator-Intent][trace={trace_id}] Enriched skill_clarification with user answer")

            if hasattr(intent, "thinking_plan") and intent.thinking_plan:
                safe_thinking_plan = orch._sanitize_intent_thinking_plan_for_skill_task(intent.thinking_plan)
                if safe_thinking_plan:
                    task_args["thinking_plan"] = safe_thinking_plan

            plan = task_args.get("thinking_plan") if isinstance(task_args.get("thinking_plan"), dict) else {}
            plan_keys = sorted(plan.keys())[:12] if plan else []
            log_info_fn(
                f"[Orchestrator-Intent][trace={trace_id}] Calling autonomous_skill_task complexity={task_args.get('complexity')} plan_keys={plan_keys}"
            )

            call_tool_async = getattr(hub, "call_tool_async", None)
            if asyncio.iscoroutinefunction(call_tool_async):
                result = await call_tool_async("autonomous_skill_task", task_args)
            else:
                result = await asyncio.to_thread(hub.call_tool, "autonomous_skill_task", task_args)

            if isinstance(result, dict):
                if result.get("success"):
                    intent.mark_executed()
                    store.update_state(intent.id, intent_state_cls.EXECUTED)
                    skill_name = result.get("skill_name", intent.skill_name)
                    exec_result = result.get("execution_result", {})
                    validation_score = result.get("validation_score", 0)
                    log_info_fn(
                        f"[Orchestrator-Intent][trace={trace_id}] Skill {skill_name} created (score: {validation_score})"
                    )
                    response_text = f"✅ Skill **{skill_name}** wurde erstellt und ausgeführt!\n\n"
                    response_text += f"**Validation Score:** {validation_score:.0%}\n\n"
                    if exec_result:
                        response_text += (
                            f"**Ergebnis:**\n```json\n{json.dumps(exec_result, indent=2, ensure_ascii=False)[:500]}\n```"
                        )
                    return core_chat_response_cls(
                        model="system",
                        content=response_text,
                        conversation_id=conversation_id,
                    )
                else:
                    if result.get("skill_created"):
                        skill_name = result.get("skill_name", intent.skill_name)
                        run_error = result.get("error", "Unbekannter Laufzeitfehler")
                        intent.mark_executed()
                        store.update_state(intent.id, intent_state_cls.EXECUTED)
                        log_warn_fn(
                            f"[Orchestrator-Intent][trace={trace_id}] Skill {skill_name} created, but first execution failed: {run_error}"
                        )
                        return core_chat_response_cls(
                            model="system",
                            content=(
                                f"✅ Skill **{skill_name}** wurde erstellt.\n\n"
                                f"⚠️ Der erste Testlauf ist fehlgeschlagen: {run_error}\n"
                                f"(trace: {trace_id})"
                            ),
                            conversation_id=conversation_id,
                        )
                    error = result.get("error", "Unknown error")
                    log_error_fn(f"[Orchestrator-Intent][trace={trace_id}] autonomous_skill_task failed: {error}")
                    intent.mark_failed()
                    store.update_state(intent.id, intent_state_cls.FAILED)
                    return core_chat_response_cls(
                        model="system",
                        content=f"❌ Skill-Erstellung fehlgeschlagen: {error} (trace: {trace_id})",
                        conversation_id=conversation_id,
                    )

            intent.mark_executed()
            store.update_state(intent.id, intent_state_cls.EXECUTED)
            return core_chat_response_cls(
                model="system",
                content="✅ Skill-Anfrage wurde verarbeitet.",
                conversation_id=conversation_id,
            )
        except Exception as exc:
            log_error_fn(f"[Orchestrator-Intent] Create failed: {exc}")
            store.update_state(intent.id, intent_state_cls.FAILED)
            return core_chat_response_cls(
                model="system",
                content=f"❌ Fehler beim Erstellen: {exc}",
                conversation_id=conversation_id,
            )
    elif is_negative:
        intent.reject()
        store.update_state(intent.id, intent_state_cls.REJECTED)
        log_info_fn(f"[Orchestrator-Intent] Skill {intent.skill_name} creation rejected")
        return core_chat_response_cls(
            model="system",
            content="❌ Skill-Erstellung abgebrochen.",
            conversation_id=conversation_id,
        )

    return None


async def process_chunked_stream(
    orch: Any,
    user_text: str,
    conversation_id: str,
    request: Any,
    *,
    get_hub_fn: Callable[[], Any],
    log_info_fn: Callable[[str], None],
    log_error_fn: Callable[[str], None],
) -> AsyncGenerator[Tuple[str, bool, Dict], None]:
    log_info_fn("[Orchestrator-Chunking] v3 MCP-basierte Analyse startet...")
    hub = get_hub_fn()

    yield ("", False, {"type": "document_analysis_start", "message": "Preprocessing document..."})

    try:
        if hasattr(hub, "call_tool_async"):
            preprocess_result = await hub.call_tool_async(
                "preprocess",
                {
                    "text": user_text,
                    "add_paragraph_ids": True,
                    "normalize_whitespace": True,
                    "remove_artifacts": True,
                },
            )
        else:
            preprocess_result = await asyncio.to_thread(
                hub.call_tool,
                "preprocess",
                {
                    "text": user_text,
                    "add_paragraph_ids": True,
                    "normalize_whitespace": True,
                    "remove_artifacts": True,
                },
            )
        processed_text = preprocess_result.get("text", user_text)
        log_info_fn(f"[Orchestrator-Chunking] Preprocessed: {len(processed_text)} chars")
    except Exception as exc:
        log_error_fn(f"[Orchestrator-Chunking] Preprocess failed: {exc}, using raw text")
        processed_text = user_text

    yield ("", False, {"type": "document_analysis_progress", "message": "Analyzing document structure..."})

    try:
        if hasattr(hub, "call_tool_async"):
            structure = await hub.call_tool_async("analyze_structure", {"text": processed_text})
        else:
            structure = await asyncio.to_thread(hub.call_tool, "analyze_structure", {"text": processed_text})
        log_info_fn(
            f"[Orchestrator-Chunking] Structure: {structure.get('heading_count', 0)} Headings, "
            f"{structure.get('code_blocks', 0)} Code-Bloecke, "
            f"Complexity {structure.get('complexity', 0)}/10"
        )
        compact_summary = orch._build_summary_from_structure(structure)
    except Exception as exc:
        log_error_fn(f"[Orchestrator-Chunking] Structure analysis failed: {exc}")
        structure = {
            "heading_count": 0,
            "code_blocks": 0,
            "complexity": 5,
            "keywords": [],
            "intro": processed_text[:500],
        }
        compact_summary = f"Text ({len(processed_text)} chars)"

    yield (
        "",
        False,
        {
            "type": "document_analysis_done",
            "structure": {
                "total_chars": structure.get("total_chars", len(processed_text)),
                "total_tokens": structure.get("total_tokens", len(processed_text) // 4),
                "total_lines": structure.get("total_lines", processed_text.count("\n")),
                "heading_count": structure.get("heading_count", 0),
                "headings": structure.get("headings", [])[:10],
                "code_blocks": structure.get("code_blocks", 0),
                "code_languages": structure.get("languages", []),
                "keywords": structure.get("keywords", []),
                "estimated_complexity": structure.get("complexity", 5),
            },
            "message": (
                f"Struktur erkannt: {structure.get('heading_count', 0)} Abschnitte, "
                f"{structure.get('code_blocks', 0)} Code-Bloecke"
            ),
        },
    )

    yield ("", False, {"type": "thinking_start", "message": "Analysiere Inhalt..."})

    analysis_prompt = f"""Analysiere folgendes Dokument anhand der Struktur-Uebersicht:

{compact_summary}

Der User hat dieses Dokument gesendet. Was ist sein wahrscheinlicher Intent?
Braucht die Antwort Sequential Thinking (schrittweises Reasoning)?"""

    thinking_result = await orch.thinking.analyze(analysis_prompt)
    log_info_fn(
        f"[Orchestrator-Chunking] ThinkingLayer: intent={thinking_result.get('intent')}, "
        f"needs_sequential={thinking_result.get('needs_sequential_thinking')}"
    )

    yield (
        "",
        False,
        {
            "type": "chunking_done",
            "conversation_id": conversation_id,
            "method": "mcp_v3",
            "aggregated_summary": compact_summary,
            "structure": {
                "headings": structure.get("headings", []),
                "keywords": structure.get("keywords", []),
                "complexity": structure.get("complexity", 5),
            },
            "thinking_result": thinking_result,
            "needs_sequential_any": thinking_result.get("needs_sequential_thinking", False)
            or thinking_result.get("sequential_thinking_required", False),
            "max_complexity": structure.get("complexity", 5),
        },
    )


async def execute_control_layer(
    orch: Any,
    user_text: str,
    thinking_plan: Dict[str, Any],
    memory_data: str,
    conversation_id: str,
    *,
    response_mode: str = "interactive",
    intent_system_available: bool,
    skill_creation_intent_cls: Any,
    intent_origin_cls: Any,
    get_intent_store_fn: Callable[[], Any],
    log_info_fn: Callable[[str], None],
    log_warn_fn: Callable[[str], None],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    skip_control, skip_reason_sync = orch._should_skip_control_layer(user_text, thinking_plan)
    if skip_control:
        log_info_fn(f"[Orchestrator] === LAYER 2: CONTROL === SKIPPED ({skip_reason_sync})")

    if thinking_plan.get("_sequential_deferred"):
        log_info_fn(
            "[Orchestrator] Sequential deferred: "
            f"{thinking_plan.get('_sequential_deferred_reason', 'interactive_mode')}"
        )

    if thinking_plan.get("needs_sequential_thinking") or thinking_plan.get("sequential_thinking_required"):
        log_info_fn("[Orchestrator] Sequential Thinking detected - executing BEFORE Control...")
        from config import get_sequential_timeout_s

        seq_timeout = get_sequential_timeout_s()
        try:
            sequential_result = await asyncio.wait_for(
                orch.control._check_sequential_thinking(user_text=user_text, thinking_plan=thinking_plan),
                timeout=float(seq_timeout),
            )
            if sequential_result:
                thinking_plan["_sequential_result"] = sequential_result
                log_info_fn(
                    f"[Orchestrator] Sequential completed: {len(sequential_result.get('steps', []))} steps"
                )
        except asyncio.TimeoutError:
            log_warn_fn(
                f"[Orchestrator] Sequential timeout after {seq_timeout}s — continuing without sequential result"
            )
            thinking_plan["_sequential_timed_out"] = True

    if skip_control:
        verified_plan = thinking_plan.copy()
        verified_plan["_verified"] = False
        verified_plan["_skipped"] = True
        verified_plan["_skip_reason"] = str(skip_reason_sync or "low_risk_skip")
        verified_plan["_final_instruction"] = ""
        verified_plan["_warnings"] = []
        verification = {
            "approved": True,
            "hard_block": False,
            "decision_class": "allow",
            "block_reason_code": "",
            "reason": "control_skipped",
            "corrections": {},
        }
        control_decision = ControlDecision.from_verification(
            verification,
            default_approved=False,
        )
    else:
        log_info_fn("[Orchestrator] === LAYER 2: CONTROL ===")
        verification = await orch.control.verify(
            user_text,
            thinking_plan,
            memory_data,
            response_mode=response_mode,
        )
        log_info_fn(f"[Orchestrator-Control] approved={verification.get('approved')}")
        log_info_fn(f"[Orchestrator-Control] warnings={verification.get('warnings', [])}")
        verified_plan = orch.control.apply_corrections(thinking_plan, verification)
        control_decision = ControlDecision.from_verification(
            verification,
            default_approved=False,
        )
        if verification.get("_needs_skill_confirmation") and intent_system_available:
            skill_name = verification.get("_skill_name", "unknown")
            log_info_fn(f"[Orchestrator] Creating SkillCreationIntent for '{skill_name}'")

            intent = skill_creation_intent_cls(
                skill_name=skill_name,
                origin=intent_origin_cls.USER,
                reason=verification.get("_cim_decision", {}).get("pattern_id", "control_layer"),
                user_text=user_text,
                conversation_id=conversation_id,
                thinking_plan=thinking_plan,
                complexity=thinking_plan.get("sequential_complexity", 5),
            )
            store = get_intent_store_fn()
            store.add(intent)
            verified_plan["_pending_intent"] = intent.to_dict()
            log_info_fn(f"[Orchestrator] Intent {intent.id[:8]} added to verified_plan")

    persist_control_decision(verified_plan, control_decision)
    persist_execution_result(verified_plan, ExecutionResult())

    from config import get_skill_auto_create_on_low_risk

    if (
        verified_plan.get("_pending_intent")
        and get_skill_auto_create_on_low_risk()
        and (thinking_plan or {}).get("hallucination_risk") == "low"
        and not verified_plan["_pending_intent"].get("needs_package_install", False)
    ):
        bypassed_skill = verified_plan["_pending_intent"].get("skill_name", "?")
        del verified_plan["_pending_intent"]
        verified_plan["_auto_create_bypass"] = True
        log_info_fn(
            f"[Orchestrator] Auto-create bypass: skill={bypassed_skill} "
            "(low_risk, no_packages, SKILL_AUTO_CREATE_ON_LOW_RISK=true)"
        )

    return verification, verified_plan
