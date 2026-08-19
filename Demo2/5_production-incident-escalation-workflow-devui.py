"""Production Incident Escalation Workflow Sample for DevUI.

A hierarchical multi-agent workflow modeled on a real engineering incident
response process, going one step further than a flat orchestrator by adding
a final rule-based escalation stage after the hierarchy resolves:

    Incident Commander Agent  (Level 0 -- triage)
            |
            | (classifies severity P1-P4, briefs each team)
            v
    ---------------------------------------------
    |               |                |
    Backend-Team    Infrastructure-   Database-Team
    Lead            Team Lead         Lead              (Level 1 -- each is
    |    |          |     |           |     |            its own mini
    specialists...  specialists...    specialists...      orchestrator)
    ---------------------------------------------
            |
            v
    RCA Synthesizer Agent            (Level 2 -- combines findings into a
            |                         root-cause-analysis draft)
            v
    Escalation Manager               (Level 3 -- applies a deterministic
                                       escalation matrix based on severity,
                                       then drafts the right stakeholder
                                       notification and produces the final
                                       incident report)

Level 0 (Incident Commander): reads the incident report, classifies
severity (P1-P4), and writes a tailored investigation brief for each of the
three engineering teams -- telling a team to dig deep or stay minimal based
on whether the incident actually implicates their area.

Level 1 (Team Leads): each team lead reads only its own brief and
independently plans -- via its own LLM call -- which of ITS specialists are
worth dispatching (it can decide none are relevant, contributing a short
"not implicated" note instead of inventing findings). Selected specialists
investigate concurrently.

Level 2 (Specialists): narrow leaf-level investigators, each focused on one
subsystem (e.g. just database replication, just auth).

Level 2.5 (RCA Synthesizer): combines all team findings into a single
root-cause-analysis draft with recommended remediation.

Level 3 (Escalation Manager): this is the "escalation matrix" step -- who
gets notified is decided by a deterministic, code-level rule keyed off the
severity the Incident Commander assigned (this mirrors how real incident
response tooling works: severity routing is a hard business rule, not left
to model judgment), and only the notification message itself is drafted by
an LLM.

Design note on robustness: the workflow graph is Chief -> fan-out to 3 team
leads -> fan-in to RCA Synthesizer -> single edge to Escalation Manager.
That's the same proven fan-out/fan-in/edge primitives already used in the
earlier samples in this project; all of the request-dependent adaptivity
(which teams matter, which specialists fire, who gets escalated to) happens
inside each executor's own Python/asyncio code and the deterministic
ESCALATION_MATRIX lookup, not via conditional graph routing.

Includes fixes already discovered while building the earlier samples in
this project:

FIXED: `WorkflowBuilder` requires `start_executor` as a keyword-only
constructor argument -- the old fluent `.set_start_executor(...)` method
was removed in the current agent_framework version.

FIXED: gpt-5 is a reasoning model. On the Responses API, hidden reasoning
tokens are billed against the same `max_output_tokens` budget as the
visible answer, so a tight budget can be entirely consumed by reasoning and
leave `response.output_text` empty. Fix: pass `reasoning={"effort": "low"}`
and an explicit `text.format`, and use generous token budgets.

FIXED: `serve()` in the current `agent_framework_devui` uses
`instrumentation_enabled`, not the older `tracing_enabled` kwarg.

To run:
    pip install --upgrade agent-framework azure-ai-projects azure-identity python-dotenv
    python 5_production-incident-escalation-workflow-devui.py
"""

import os
import json
import asyncio
import logging
import random
from typing import Any
from dotenv import load_dotenv
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    WorkflowViz,
)
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from agent_framework.devui import serve

# Configuration is kept outside the source code in .env. This avoids hard-coding
# deployment-specific values and lets the workflow run in other environments.
load_dotenv()
project_endpoint = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT") or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
model = os.getenv("AI_FOUNDRY_DEPLOYMENT_NAME") or os.getenv("MODEL_DEPLOYMENT_NAME")
azure_tenant_id = os.getenv("AZURE_TENANT_ID")

print("Project Endpoint: ", project_endpoint)
print("Model: ", model)

if not project_endpoint or not model:
    missing = []
    if not project_endpoint:
        missing.append("AI_FOUNDRY_PROJECT_ENDPOINT or FOUNDRY_PROJECT_ENDPOINT")
    if not model:
        missing.append("AI_FOUNDRY_DEPLOYMENT_NAME or MODEL_DEPLOYMENT_NAME")
    raise ValueError("Missing required .env value(s): " + ", ".join(missing))


async def run_agent_with_retry(agent: Any, message, *, max_tokens: int = 800):
    """Run an agent and wait/retry when Azure returns a transient rate limit."""
    max_attempts = int(os.getenv("AGENT_RETRY_ATTEMPTS", "5"))
    for attempt in range(max_attempts):
        try:
            return await agent.run(message, max_tokens=max_tokens)
        except Exception as exc:
            error_text = str(exc).lower()
            is_rate_limit = (
                "429" in error_text
                or "too many requests" in error_text
                or "rate_limit" in error_text
                or "rate limit" in error_text
            )
            if not is_rate_limit or attempt == max_attempts - 1:
                raise

            delay = min(30, (2 ** attempt) + random.uniform(0.25, 1.25))
            print(f"Rate limit hit. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)


class LazyFoundryAgent:
    """Calls the Foundry Responses API from the DevUI request loop."""

    def __init__(self, agent_name: str, agent_instructions: str):
        self.agent_name = agent_name
        self.agent_instructions = agent_instructions
        self.credential: AzureCliCredential | None = None
        self.project_client: AIProjectClient | None = None
        self.openai_client: Any | None = None

    async def _ensure_client(self):
        if self.openai_client is None:
            self.credential = (
                AzureCliCredential(tenant_id=azure_tenant_id)
                if azure_tenant_id
                else AzureCliCredential()
            )
            self.project_client = AIProjectClient(
                endpoint=project_endpoint,
                credential=self.credential,
            )
            self.openai_client = self.project_client.get_openai_client()
            print(f"{self.agent_name} client initialized in the DevUI event loop.")

    async def run(self, message: str, *, max_tokens: int, json_mode: bool = False):
        await self._ensure_client()

        text_format: dict[str, Any] = {"type": "json_object"} if json_mode else {"type": "text"}

        response = await self.openai_client.responses.create(
            model=model,
            # This deployment returns empty content when the Responses API's
            # `instructions` field is used. Put the role guidance in the input
            # instead, which is supported reliably by the deployed model.
            input=f"Instructions:\n{self.agent_instructions}\n\nUser request:\n{message}",
            max_output_tokens=max_tokens,
            # gpt-5 is a reasoning model: hidden reasoning tokens are billed
            # against max_output_tokens and can consume the entire budget,
            # leaving nothing for the visible answer. Keep effort low and
            # force a message format so output_text is reliably non-empty.
            reasoning={"effort": "low"},
            text={"format": text_format},
        )
        if not response.output_text:
            status = getattr(response, "status", "unknown")
            incomplete = getattr(response, "incomplete_details", None)
            raise RuntimeError(
                f"{self.agent_name} returned an empty response "
                f"(status={status}, incomplete_details={incomplete}, max_tokens={max_tokens})"
            )
        return response.output_text


def create_agent(agent_name: str, agent_instructions: str) -> LazyFoundryAgent:
    """Return a lazily initialized Foundry agent."""
    return LazyFoundryAgent(agent_name, agent_instructions)


def _strip_code_fences(raw_text: str) -> str:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if "\n" in cleaned:
            first_line, rest = cleaned.split("\n", 1)
            if first_line.strip().lower() in ("json", ""):
                cleaned = rest
    return cleaned.strip()


# ---------------------------------------------------------------------------
# Level 0: Incident Commander
# ---------------------------------------------------------------------------
# Expected JSON shape:
# {
#   "severity": "P1",
#   "summary": "Checkout API returning 500s for ~15% of requests",
#   "team_briefs": {
#     "backend": "...",
#     "infrastructure": "...",
#     "database": "..."
#   }
# }

TEAM_NAMES = ["backend", "infrastructure", "database"]
VALID_SEVERITIES = {"P1", "P2", "P3", "P4"}

DEFAULT_TEAM_BRIEFS = {
    "backend": "Investigate application-layer causes at a normal level of depth.",
    "infrastructure": "Investigate infra/network/deploy causes at a normal level of depth.",
    "database": "Investigate database causes at a normal level of depth.",
}

# This is the actual escalation matrix: a deterministic business rule, not
# something left to an LLM to decide. Severity routing needs to be
# consistent and auditable, so it's plain code -- only the notification
# text itself is drafted by an agent.
ESCALATION_MATRIX = {
    "P1": {
        "notify": "VP of Engineering + Customer Success (immediate page)",
        "sla": "Acknowledge within 5 minutes; status update every 15 minutes.",
    },
    "P2": {
        "notify": "Engineering Director",
        "sla": "Acknowledge within 30 minutes; status update every hour.",
    },
    "P3": {
        "notify": "Engineering Manager",
        "sla": "Acknowledge within 2 hours; daily status update until resolved.",
    },
    "P4": {
        "notify": "No escalation beyond the responding team lead(s).",
        "sla": "Handle during normal business hours; no forced status cadence.",
    },
}


def _parse_commander_plan(raw_text: str) -> dict[str, Any]:
    cleaned = _strip_code_fences(raw_text)
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Incident Commander returned unparseable JSON: {exc}\nRaw text:\n{raw_text}"
        ) from exc

    severity = str(plan.get("severity", "")).strip().upper()
    if severity not in VALID_SEVERITIES:
        print(f"Commander returned unrecognized severity {severity!r}; defaulting to P3.")
        severity = "P3"

    summary = str(plan.get("summary", "")).strip() or "No summary provided."
    raw_briefs = plan.get("team_briefs") or {}
    team_briefs = {
        team: str(raw_briefs.get(team, "")).strip() or DEFAULT_TEAM_BRIEFS[team]
        for team in TEAM_NAMES
    }
    return {"severity": severity, "summary": summary, "team_briefs": team_briefs}


class IncidentCommanderExecutor(Executor):
    """Level 0. Triages the incident: assigns severity and writes a
    tailored investigation brief for each downstream team. Every team lead
    receives this same message via a static fan-out edge; each reads out
    only its own brief."""

    def __init__(self, commander_agent: LazyFoundryAgent, **kwargs):
        super().__init__(**kwargs)
        self.commander_agent = commander_agent

    @handler
    async def handle(self, incident_report: str, ctx: WorkflowContext[dict[str, Any]]) -> None:
        triage_prompt = (
            f"Incident report:\n{incident_report}\n\n"
            "Classify the severity as one of P1 (critical, major customer impact, "
            "all hands), P2 (high, significant impact, urgent), P3 (moderate, "
            "limited impact, business hours), or P4 (low, minor/cosmetic, no "
            "urgency). Write a one-sentence summary. Then write a short (1-2 "
            "sentence) investigation brief for each of three engineering teams, "
            "based on what the report actually implicates. If a team's area isn't "
            "implicated at all, say so and tell them to stay minimal rather than "
            "inventing findings. Teams: 'backend' (application/API/auth logic), "
            "'infrastructure' (network, cloud, deployments, config), 'database' "
            "(query performance, replication, storage). "
            'Respond with ONLY JSON in this exact shape, no prose, no markdown fences: '
            '{"severity": "P1", "summary": "...", "team_briefs": {"backend": "...", '
            '"infrastructure": "...", "database": "..."}}'
        )
        raw_plan = await run_agent_with_retry(self.commander_agent, triage_prompt, max_tokens=700)
        plan = _parse_commander_plan(str(raw_plan))
        print(f"Incident Commander triage: severity={plan['severity']}, summary={plan['summary']!r}")
        for team, brief in plan["team_briefs"].items():
            print(f"  [{team}] brief: {brief}")

        await ctx.send_message({
            "incident_report": incident_report,
            "severity": plan["severity"],
            "summary": plan["summary"],
            "team_briefs": plan["team_briefs"],
        })


# ---------------------------------------------------------------------------
# Level 1: Team Leads
# ---------------------------------------------------------------------------
# Each team lead plans its own specialist tasks. Expected JSON shape:
# {"tasks": [{"specialist": "auth", "prompt": "..."}]}
# An empty "tasks" list is valid and means this team judged itself not
# implicated in this particular incident.

def _parse_team_plan(raw_text: str, valid_specialists: set[str]) -> list[dict[str, str]]:
    cleaned = _strip_code_fences(raw_text)
    try:
        plan = json.loads(cleaned)
    except json.JSONDecodeError:
        # Likely truncated mid-string because the model ran out of tokens.
        # Try a couple of cheap repairs before giving up: close an
        # unterminated string, then balance any unclosed brackets/braces.
        repaired = cleaned
        if repaired.count('"') % 2 == 1:
            repaired += '"'
        repaired += "]" * (repaired.count("[") - repaired.count("]"))
        repaired += "}" * (repaired.count("{") - repaired.count("}"))
        try:
            plan = json.loads(repaired)
            print("Team lead JSON was truncated; auto-repaired successfully.")
        except json.JSONDecodeError:
            # Still broken -- degrade gracefully instead of failing the
            # whole workflow run. This team simply contributes no tasks.
            print(
                "Team lead returned unparseable JSON even after repair attempt; "
                f"treating as no tasks selected. Raw text:\n{raw_text}"
            )
            return []

    tasks = [
        task for task in plan.get("tasks", [])
        if isinstance(task, dict)
        and task.get("specialist") in valid_specialists
        and task.get("prompt")
    ]
    return tasks


class TeamLeadExecutor(Executor):
    """Level 1. Reads its own brief out of the shared message from the
    Commander, plans which of its specialists are relevant, dispatches them
    concurrently, and reports a team finding upward. Each subclass just
    supplies its team name, lead agent, and specialist pool."""

    def __init__(
        self,
        team_name: str,
        team_label: str,
        lead_agent: LazyFoundryAgent,
        specialist_agents: dict[str, LazyFoundryAgent],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.team_name = team_name
        self.team_label = team_label
        self.lead_agent = lead_agent
        self.specialist_agents = specialist_agents

    @handler
    async def handle(self, incident_context: dict[str, Any], ctx: WorkflowContext[dict[str, Any]]) -> None:
        brief = incident_context["team_briefs"].get(self.team_name, DEFAULT_TEAM_BRIEFS[self.team_name])
        incident_report = incident_context["incident_report"]
        severity = incident_context["severity"]
        specialist_names = sorted(self.specialist_agents.keys())

        plan_prompt = (
            f"Incident report:\n{incident_report}\n\n"
            f"Severity: {severity}\n\n"
            f"Your team's brief from the incident commander:\n{brief}\n\n"
            f"Your available specialists: {', '.join(specialist_names)}. "
            "Decide which of them are actually worth dispatching for this brief -- "
            "it is fine to select none if the brief says this area isn't implicated. "
            "For each specialist you select, write ONE short sub-prompt (max 20 words) "
            "telling them exactly what to investigate -- keep it brief so the JSON "
            "response stays short and complete. "
            'Respond with ONLY JSON in this exact shape, no prose, no markdown fences: '
            '{"tasks": [{"specialist": "<name>", "prompt": "..."}]}'
        )
        raw_team_plan = await run_agent_with_retry(self.lead_agent, plan_prompt, max_tokens=900)
        tasks = _parse_team_plan(str(raw_team_plan), set(specialist_names))
        print(f"[{self.team_label}] dispatched specialists: {[t['specialist'] for t in tasks] or 'none'}")

        async def run_specialist(task: dict[str, str]) -> dict[str, str]:
            agent = self.specialist_agents[task["specialist"]]
            prompt = (
                f"Incident report:\n{incident_report}\n\n"
                f"Severity: {severity}\n\n"
                f"Focus for this investigation:\n{task['prompt']}\n\n"
                "Give your most likely root-cause hypothesis, the evidence or "
                "signal that would confirm it, and a concrete next step or fix."
            )
            result = await run_agent_with_retry(agent, prompt, max_tokens=600)
            return {"specialist": task["specialist"], "result": str(result)}

        if tasks:
            specialist_results = await asyncio.gather(*(run_specialist(t) for t in tasks))
            team_report = "\n\n".join(
                f"{item['specialist'].capitalize()}: {item['result']}" for item in specialist_results
            )
        else:
            team_report = f"({self.team_label} determined this incident does not implicate their area.)"

        await ctx.send_message({
            "team": self.team_name,
            "team_label": self.team_label,
            "report": team_report,
            # Threaded through so the Escalation Manager can apply the
            # correct row of ESCALATION_MATRIX later -- every team lead
            # forwards the same severity it received from the commander.
            "severity": severity,
        })


# ---------------------------------------------------------------------------
# Level 2: RCA Synthesizer
# ---------------------------------------------------------------------------

class RCASynthesizerExecutor(Executor):
    """Combines every team's findings into a root-cause-analysis draft.
    Fan-in delivers a list containing one result dict per team lead. This is
    NOT the final node -- it forwards to the Escalation Manager rather than
    yielding output, since severity routing still needs to happen."""

    def __init__(self, agent: LazyFoundryAgent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, team_results: list[dict[str, Any]], ctx: WorkflowContext[dict[str, Any]]) -> None:
        # Every team lead forwards the same severity it received from the
        # commander, so any one result carries it through to this stage.
        severity = team_results[0]["severity"] if team_results else "P3"

        delay_seconds = int(os.getenv("RCA_SYNTH_DELAY_SECONDS", "5"))
        if delay_seconds > 0:
            print(f"Waiting {delay_seconds}s before RCA synthesis to avoid rate limits...")
            await asyncio.sleep(delay_seconds)

        team_reports = "\n\n".join(
            f"=== {item['team_label']} ===\n{item['report']}" for item in team_results
        ) or "(No team findings were gathered.)"

        rca_prompt = (
            f"Team findings:\n{team_reports}\n\n"
            "Write a concise root-cause-analysis draft: likely root cause (or "
            "leading hypotheses if not yet confirmed), impact, and recommended "
            "remediation steps in priority order. If a team reported their area "
            "wasn't implicated, don't include a section for it."
        )
        rca_draft = await run_agent_with_retry(self.agent, rca_prompt, max_tokens=900)

        await ctx.send_message({
            "team_results": team_results,
            "rca_draft": str(rca_draft),
            "severity": severity,
        })


# ---------------------------------------------------------------------------
# Level 3: Escalation Manager
# ---------------------------------------------------------------------------

class EscalationManagerExecutor(Executor):
    """Applies the deterministic escalation matrix based on severity, then
    has an agent draft the appropriate stakeholder notification. Produces
    the final combined incident report."""

    def __init__(self, notifier_agent: LazyFoundryAgent, **kwargs):
        super().__init__(**kwargs)
        self.notifier_agent = notifier_agent

    @handler
    async def handle(self, rca_context: dict[str, Any], ctx: WorkflowContext[str]) -> None:
        # Severity is threaded all the way from the Incident Commander,
        # through every team lead, through the RCA synthesizer. Default to
        # P3 only as a last-resort safety net if it's ever missing.
        severity = rca_context.get("severity", "P3")
        routing = ESCALATION_MATRIX.get(severity, ESCALATION_MATRIX["P3"])

        print(f"Escalation matrix lookup: severity={severity} -> notify={routing['notify']}")

        notify_prompt = (
            f"Severity: {severity}\n"
            f"Who must be notified per policy: {routing['notify']}\n"
            f"Required response SLA: {routing['sla']}\n\n"
            f"Root cause analysis draft:\n{rca_context['rca_draft']}\n\n"
            "Draft a short, professional incident notification message for the "
            "people/roles listed above. Include severity, a one-line impact "
            "summary, current status, and the SLA. If the policy says no "
            "escalation beyond the team lead is needed, instead write a short "
            "internal team-channel update rather than an escalation message."
        )
        notification_draft = await run_agent_with_retry(self.notifier_agent, notify_prompt, max_tokens=500)

        final_report = (
            f"SEVERITY: {severity}\n"
            f"ESCALATION: {routing['notify']}\n"
            f"SLA: {routing['sla']}\n\n"
            f"--- ROOT CAUSE ANALYSIS ---\n{rca_context['rca_draft']}\n\n"
            f"--- NOTIFICATION DRAFT ---\n{notification_draft}"
        )
        # yield_output marks this value as the workflow's final result.
        await ctx.yield_output(final_report)


def build_workflow():
    """Wire up the full hierarchy: Commander -> [3 Team Leads] -> RCA
    Synthesizer -> Escalation Manager. The graph is fan-out -> fan-in ->
    plain edge (all proven WorkflowBuilder primitives); the request-
    dependent adaptivity (which teams matter, which specialists fire, who
    gets escalated to) lives inside the executors and the deterministic
    ESCALATION_MATRIX, not in conditional graph routing."""

    commander_agent = create_agent(
        agent_name="Incident-Commander-Agent",
        agent_instructions=(
            "You are the incident commander for a production engineering "
            "organization. Given an incident report, classify severity and write "
            "a tailored investigation brief for each of three downstream teams. "
            "Always respond with strict JSON only, matching the schema given in "
            "the user message -- no prose, no markdown fences."
        ),
    )

    backend_specialists = {
        "api": create_agent(
            "API-Specialist-Agent",
            "You are a backend API reliability engineer. Given an incident and a "
            "specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
        "auth": create_agent(
            "Auth-Specialist-Agent",
            "You are an authentication/authorization systems engineer. Given an "
            "incident and a specific investigation focus, give your most likely "
            "root-cause hypothesis, supporting evidence, and a concrete next step "
            "for exactly that focus. Keep the answer under 150 words.",
        ),
        "integration": create_agent(
            "Integration-Specialist-Agent",
            "You are an integrations/third-party API engineer. Given an incident "
            "and a specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
    }

    infrastructure_specialists = {
        "network": create_agent(
            "Network-Specialist-Agent",
            "You are a network reliability engineer. Given an incident and a "
            "specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
        "cloud": create_agent(
            "Cloud-Infra-Specialist-Agent",
            "You are a cloud infrastructure engineer. Given an incident and a "
            "specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
        "deployment": create_agent(
            "Deployment-Specialist-Agent",
            "You are a CI/CD and deployment engineer. Given an incident and a "
            "specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
    }

    database_specialists = {
        "performance": create_agent(
            "DB-Performance-Specialist-Agent",
            "You are a database performance engineer. Given an incident and a "
            "specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
        "replication": create_agent(
            "DB-Replication-Specialist-Agent",
            "You are a database replication/HA engineer. Given an incident and a "
            "specific investigation focus, give your most likely root-cause "
            "hypothesis, supporting evidence, and a concrete next step for exactly "
            "that focus. Keep the answer under 150 words.",
        ),
    }

    backend_lead_agent = create_agent(
        "Backend-Team-Lead-Agent",
        "You lead the backend engineering team (API, auth, integrations). Given a "
        "brief, decide which of your specialists are worth dispatching and "
        "respond with strict JSON only, matching the schema given in the user "
        "message.",
    )
    infrastructure_lead_agent = create_agent(
        "Infrastructure-Team-Lead-Agent",
        "You lead the infrastructure team (network, cloud, deployments). Given a "
        "brief, decide which of your specialists are worth dispatching and "
        "respond with strict JSON only, matching the schema given in the user "
        "message.",
    )
    database_lead_agent = create_agent(
        "Database-Team-Lead-Agent",
        "You lead the database team (performance, replication/HA). Given a "
        "brief, decide which of your specialists are worth dispatching and "
        "respond with strict JSON only, matching the schema given in the user "
        "message.",
    )

    rca_synthesizer_agent = create_agent(
        "RCA-Synthesizer-Agent",
        "You are a senior site reliability engineer writing a root-cause-"
        "analysis draft. Combine the provided team findings into one clear, "
        "well-organized RCA with prioritized remediation steps.",
    )

    escalation_notifier_agent = create_agent(
        "Escalation-Notification-Agent",
        "You are an incident communications lead. Draft clear, professional "
        "stakeholder notifications for production incidents based on the "
        "severity and escalation policy you are given.",
    )

    commander_executor = IncidentCommanderExecutor(commander_agent, id="IncidentCommander")
    backend_lead_executor = TeamLeadExecutor(
        "backend", "Backend Team", backend_lead_agent, backend_specialists,
        id="BackendTeamLead",
    )
    infrastructure_lead_executor = TeamLeadExecutor(
        "infrastructure", "Infrastructure Team", infrastructure_lead_agent, infrastructure_specialists,
        id="InfrastructureTeamLead",
    )
    database_lead_executor = TeamLeadExecutor(
        "database", "Database Team", database_lead_agent, database_specialists,
        id="DatabaseTeamLead",
    )
    rca_synthesizer_executor = RCASynthesizerExecutor(rca_synthesizer_agent, id="RCASynthesizer")
    escalation_manager_executor = EscalationManagerExecutor(
        escalation_notifier_agent, id="EscalationManager"
    )

    team_leads = [backend_lead_executor, infrastructure_lead_executor, database_lead_executor]

    # NOTE: as of the current agent_framework, start_executor is a required
    # keyword-only constructor argument on WorkflowBuilder -- the old fluent
    # .set_start_executor(...) method has been removed in favor of this.
    workflow = (
        WorkflowBuilder(
            name="Production Incident Escalation Workflow",
            description=(
                "An incident commander triages and briefs three engineering team "
                "leads, each of which plans and dispatches its own specialists, "
                "before an RCA synthesizer and a rule-based escalation manager "
                "produce the final incident report."
            ),
            start_executor=commander_executor,
        )
        # Fan-out: every team lead receives the commander's combined message
        # and reads out only the brief meant for it.
        .add_fan_out_edges(commander_executor, team_leads)
        # Fan-in: the RCA synthesizer waits for all three team findings.
        .add_fan_in_edges(team_leads, rca_synthesizer_executor)
        # Sequential final stage: escalation routing happens only after the
        # RCA draft exists.
        .add_edge(rca_synthesizer_executor, escalation_manager_executor)
        .build()
    )

    # Mermaid text makes the workflow graph easy to inspect or document.
    viz = WorkflowViz(workflow)
    mermaid_content = viz.to_mermaid()
    print("Mermaid Diagram:\n", mermaid_content)

    return workflow


def _call_serve_compatibly(**desired_kwargs):
    """Call agent_framework.devui.serve() with only the kwargs it currently
    supports, dropping any it doesn't recognize (instead of crashing).

    agent_framework_devui is still preview-grade and its serve() signature
    has been changing release to release -- this keeps the script working
    across those changes without needing a patch every time one kwarg shifts.
    """
    import inspect

    supported = set(inspect.signature(serve).parameters)
    accepted = {k: v for k, v in desired_kwargs.items() if k in supported}
    dropped = set(desired_kwargs) - set(accepted)
    if dropped:
        print(f"Note: serve() in your installed agent_framework_devui doesn't "
              f"support these kwargs -- skipping them: {sorted(dropped)}")
    return serve(**accepted)


def main():
    """Launch the production incident escalation workflow in DevUI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    devui_port = int(os.getenv("DEVUI_INCIDENT_PORT") or os.getenv("DEVUI_PORT", "8095"))
    logger.info("Starting Production Incident Escalation Workflow")
    logger.info("Available at: http://localhost:%s", devui_port)
    logger.info("Entity ID: workflow_incident_escalation")

    workflow = build_workflow()
    # DevUI provides an interactive browser client. instrumentation_enabled
    # records each node and agent call so the execution can be inspected
    # (this kwarg was renamed from `tracing_enabled` in the current
    # agent_framework_devui version).
    _call_serve_compatibly(
        entities=[workflow], port=devui_port, auto_open=True, instrumentation_enabled=True
    )


if __name__ == "__main__":
    main()


#######Suggested - user prompts

#Users are seeing intermittent 500 errors on the checkout API for the last 20 minutes, affecting about 15% of requests.

#Database replication lag has grown to 45 minutes on the primary read replica, reporting dashboards are showing stale data.

#Customers report they cannot log in via SSO since this morning's deployment; error rate is around 80%.

#Elevated latency (2-3x normal) on the recommendations service since a config change 10 minutes ago, no errors yet, low customer impact so far.
