"""Extract methodological details from research papers using multi-turn LLM.

Uses OpenAI's responses API with file uploads for PDF processing.
Three-turn extraction: extract details -> find missed info -> synthesize protocol.
"""

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from paimon import cfg


TURN_1_TEMPLATE = """\
Extract every methodological detail for reproducing {task} from this paper.
For each parameter or choice:
- Exact value/setting (or mark as "not specified")
- Contextual note if needed for correct interpretation

Include decisions mentioned only briefly. Do not summarize—extract.
"""

TURN_2_TEMPLATE = """\
You are now a skeptical reproducer, not a reader. Assume you must run
this task tomorrow on your own machine.
Review the paper again and identify any procedural information you did NOT
include in Turn 1. Focus on:
- Steps that seem "obvious" but are actually specified
- Conditional branches (if X then Y)
- Negative constraints (what NOT to do)
- Implicit sequencing (A must precede B)
- Inferable: not explicitely stated, but should forced by other stated choices, or standard within the exact sub-method used. In your report, note the condition is inferred.
- Constraints: what is held fixed, restricted, forbidden, or allowed during the simulation. Methods with the same name can differ fundamentally depending on their constraints (e.g. a relaxation under fixed symmetry (in crystal) bond length (in molecule), cell shape, or atom subset).

List only the additions.
"""

TURN_3_TEMPLATE = """\
Synthesize Turns 1-2 into a self-contained simulation protocol formatted
in Markdown. Write as if the reader has NO access to the original paper.

Structure requirements:

## Top-level sections (##): Major workflow phases
Organize by the natural execution or conceptual flow of THIS methodology.
Common patterns (adapt as needed for this paper):
- Preparation -> Execution -> Analysis
- Model Setup -> Calculation -> Post-processing
- Initialization -> Sampling -> Observable Extraction

## Subsections (###): Logical groupings
Create subsections when switching between:
- Distinct physical quantities or computational controls
- Conceptually separate operations
- Significant methodological changes
- Procedural branching or conditional logic

## Numbered steps: Sequential actions
Within each subsection, use numbered lists (1., 2., 3.) for ordered steps.

## Software Implementation Notes (final section)
For each software:
- List steps where used (by section/subsection number)
- Include versions, settings, file names if specified
- Flag unavailable tools: METHOD_UNAVAILABLE

Replace all paper references with actual content:
- Equations: write them out
- Tables/Figures: embed relevant values
- External citations: note as METHOD_UNAVAILABLE if details absent

Structure should reflect this paper's specific workflow, not a template.
Omit phases that don't apply. Use imperative tone throughout. Do NOT
summarize or compress—preserve all specificity from Turns 1-2.
"""


@dataclass
class ExtractionResult:
    """Result of paper extraction."""

    turn1_response: str
    turn2_response: str
    turn3_response: str
    protocol: str  # alias for turn3_response


def _parse_model_spec(model_spec: str) -> str:
    """Parse model specification to get the model name.

    Supports paimon's "{api}/{model}" format.
    Only OpenAI models are supported for this extraction.
    """
    parts = model_spec.split("/")
    if len(parts) == 1:
        return model_spec
    api, model = parts[0], "/".join(parts[1:])
    if api != "openai":
        raise ValueError(
            f"Only OpenAI models supported for paper extraction, got: {api}"
        )
    return model


def extract_methodology(
    paper_path: str | Path,
    task: str,
    *,
    supporting_info_path: str | Path | None = None,
    model: str | None = None,
    reasoning_effort: str = "medium",
    verbosity: str = "medium",
) -> ExtractionResult:
    """Extract methodological details from a research paper.

    Uses a three-turn conversation with OpenAI's responses API:
    1. Extract all methodological details
    2. Review for missed procedural information
    3. Synthesize into a self-contained simulation protocol

    Parameters
    ----------
    paper_path
        Path to the main paper PDF file.
    task
        Description of what to extract (e.g., "diffusivities of liquid
        electrolytes in Li-ion batteries").
    supporting_info_path
        Optional path to supporting information PDF.
    model
        LLM model specification. Supports paimon's "{api}/{model}" format.
        Defaults to cfg.base_reasoning_llm. Only OpenAI models are supported.
    reasoning_effort
        OpenAI reasoning effort: "low", "medium", or "high".
    verbosity
        OpenAI text verbosity: "low", "medium", or "high".

    Returns
    -------
    ExtractionResult
        Contains responses from all three turns and the final protocol.
    """
    model = model or cfg.base_reasoning_llm
    model_name = _parse_model_spec(model)

    client = OpenAI()

    paper_path = Path(paper_path)
    if not paper_path.exists():
        raise FileNotFoundError(f"Paper not found: {paper_path}")

    with open(paper_path, "rb") as f:
        paper_file = client.files.create(file=f, purpose="user_data")

    initial_content: list[dict] = [
        {"type": "input_text", "text": "This is a research article."},
        {"type": "input_file", "file_id": paper_file.id},
    ]

    if supporting_info_path:
        supporting_info_path = Path(supporting_info_path)
        if not supporting_info_path.exists():
            raise FileNotFoundError(
                f"Supporting info not found: {supporting_info_path}"
            )
        with open(supporting_info_path, "rb") as f:
            si_file = client.files.create(file=f, purpose="user_data")
        initial_content.extend(
            [
                {
                    "type": "input_text",
                    "text": "This is a supporting information of the article.",
                },
                {"type": "input_file", "file_id": si_file.id},
            ]
        )

    reasoning_opts = {"effort": reasoning_effort, "summary": "detailed"}
    text_opts = {"verbosity": verbosity}

    turn1_prompt = TURN_1_TEMPLATE.format(task=task)
    initial_content.append({"type": "input_text", "text": turn1_prompt})

    r1 = client.responses.create(
        model=model_name,
        store=cfg.open_ai_store,
        reasoning=reasoning_opts,  # type: ignore
        text=text_opts,  # type: ignore
        input=[{"role": "user", "content": initial_content}],  # type: ignore
    )

    turn2_content = [{"type": "input_text", "text": TURN_2_TEMPLATE}]
    r2 = client.responses.create(
        model=model_name,
        previous_response_id=r1.id,
        store=cfg.open_ai_store,
        reasoning=reasoning_opts,  # type: ignore
        text=text_opts,  # type: ignore
        input=[{"role": "user", "content": turn2_content}],  # type: ignore
    )

    turn3_content = [{"type": "input_text", "text": TURN_3_TEMPLATE}]
    r3 = client.responses.create(
        model=model_name,
        previous_response_id=r2.id,
        store=cfg.open_ai_store,
        reasoning=reasoning_opts,  # type: ignore
        text=text_opts,  # type: ignore
        input=[{"role": "user", "content": turn3_content}],  # type: ignore
    )

    def get_text(response) -> str:
        for item in response.output:
            if item.type == "message":
                for content in item.content:
                    if content.type == "output_text":
                        return content.text
        return ""

    turn1_text = get_text(r1)
    turn2_text = get_text(r2)
    turn3_text = get_text(r3)

    return ExtractionResult(
        turn1_response=turn1_text,
        turn2_response=turn2_text,
        turn3_response=turn3_text,
        protocol=turn3_text,
    )
